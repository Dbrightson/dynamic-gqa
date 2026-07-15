#!/usr/bin/env python3
"""Alignment experiment for Llama-style attention (k_proj/v_proj modules):
Qwen3-4B, Llama-3.x, Gemma-2/3 (text). Same protocol as align_experiment.py:
fit closed-form ridge maps between KV heads, measure aligned cosine on
held-out tokens, simulate leader+adapter sharing vs naive mean merging.
"""
import argparse, json, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_texts(split):
    from datasets import load_dataset
    out = {}
    def add(name, fn):
        try: out[name] = fn(); print(f"[data] loaded '{name}'", flush=True)
        except Exception as e: print(f"[data] SKIP {name}: {e}", flush=True)
    if split == "collect":
        add("prose", lambda: "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]))
        add("math", lambda: "\n\n".join(r["question"] + "\n" + r["answer"] for r in load_dataset("openai/gsm8k", "main", split="test")))
        add("code", lambda: "\n\n".join(r["code"] for r in load_dataset("google-research-datasets/mbpp", split="test")))
        add("stories", lambda: "\n\n".join(r["text"] for r in load_dataset("roneneldan/TinyStories", split="validation")))
    else:
        add("prose", lambda: "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")["text"]))
        add("math", lambda: "\n\n".join(r["question"] + "\n" + r["answer"] for r in load_dataset("openai/gsm8k", "main", split="train").select(range(400))))
        add("code", lambda: "\n\n".join(r["code"] for r in load_dataset("google-research-datasets/mbpp", split="train")))
        add("stories", lambda: "\n\n".join(r["text"] for r in load_dataset("roneneldan/TinyStories", split="train[:2000]")))
    return out


def find_layers(model):
    for path in ["model.layers", "model.language_model.layers",
                 "language_model.model.layers"]:
        obj = model
        try:
            for part in path.split("."): obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise RuntimeError("could not locate decoder layers")


class LlamaKV:
    """record/share hooks on k_proj and v_proj outputs (pre-RoPE)."""
    def __init__(self, layers, Hkv, dh):
        self.Hkv, self.dh = Hkv, dh
        self.mode = "off"
        self.k, self.v = {}, {}
        self.plan = None
        self.naive = None
        for i, layer in enumerate(layers):
            layer.self_attn.k_proj.register_forward_hook(self._mk(i, "k"))
            layer.self_attn.v_proj.register_forward_hook(self._mk(i, "v"))
    def _mk(self, li, which):
        def hook(mod, inp, outp):
            if self.mode == "off": return outp
            shp = outp.shape
            x = outp.view(*shp[:-1], self.Hkv, self.dh)
            if self.mode == "record":
                (self.k if which == "k" else self.v)[li] = x.detach()
                return outp
            x = x.clone()
            if self.mode == "share":
                for (ld, mb, Rk, Rv) in self.plan[li]:
                    R = Rk if which == "k" else Rv
                    x[..., mb, :] = x[..., ld, :] @ R
            elif self.mode == "naive":
                for members in self.naive[li]:
                    if len(members) > 1:
                        x[..., members, :] = x[..., members, :].mean(-2, keepdim=True)
            return x.view(shp)
        return hook


def fit_layer(Xtr, Xte, lam=1e-3):
    N, H, dh = Xtr.shape
    dev = Xtr.device
    C = torch.einsum("nga,nhb->ghab", Xtr, Xtr) / N
    eye = torch.eye(dh, device=dev)
    sim = torch.zeros(H, H)
    Rd = {}
    for g in range(H):
        A = C[g, g] + lam * eye
        R = torch.linalg.solve(A, C[g])
        pred = torch.einsum("na,hab->nhb", Xte[:, g, :], R)
        pn = pred / (pred.norm(dim=-1, keepdim=True) + 1e-9)
        tn = Xte / (Xte.norm(dim=-1, keepdim=True) + 1e-9)
        sim[g] = (pn * tn).sum(-1).mean(0).cpu()
        Rd[g] = R.cpu()
    return sim, Rd


def cluster(sim, G):
    from sklearn.cluster import AgglomerativeClustering
    G = min(G, sim.shape[0])
    s = np.minimum(sim, sim.T)
    dist = np.clip(1.0 - s, 0, None)
    np.fill_diagonal(dist, 0.0)
    return AgglomerativeClustering(n_clusters=G, metric="precomputed",
                                   linkage="average").fit_predict(dist).tolist()


def labels_to_groups(labels):
    g = {}
    for h, l in enumerate(labels): g.setdefault(l, []).append(h)
    return list(g.values())


@torch.no_grad()
def run_pass(model, ids, seq_len, device, stored=None):
    nll, ntok, agree = 0.0, 0, 0
    new_store = []
    for c in range(len(ids) // seq_len):
        chunk = ids[c*seq_len:(c+1)*seq_len].unsqueeze(0).to(device)
        out = model(chunk, labels=chunk)
        T = chunk.shape[1] - 1
        nll += out.loss.item() * T; ntok += T
        am = out.logits[0, :-1].float().argmax(-1).cpu()
        if stored is None: new_store.append(am)
        else: agree += (am == stored[c]).sum().item()
    ppl = float(np.exp(nll / max(ntok, 1)))
    if stored is None: return ppl, new_store
    return ppl, agree / max(ntok, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--out", default="results_align_qwen")
    ap.add_argument("--fit-tokens", type=int, default=12288)
    ap.add_argument("--test-tokens", type=int, default=4096)
    ap.add_argument("--eval-tokens", type=int, default=4096)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--groups", default="2,4")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    Gs = [int(g) for g in args.groups.split(",")]
    dev = args.device

    from sklearn.metrics import adjusted_rand_score
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32,
        device_map=dev)
    model.eval()
    cfg = model.config
    if hasattr(cfg, "text_config"): cfg = cfg.text_config
    Hq = cfg.num_attention_heads
    Hkv = getattr(cfg, "num_key_value_heads", Hq)
    dh = getattr(cfg, "head_dim", None) or cfg.hidden_size // Hq
    layers = find_layers(model)
    L = len(layers)
    print(f"[model] {args.model}: L={L} Hq={Hq} Hkv={Hkv} dh={dh}", flush=True)
    kv = LlamaKV(layers, Hkv, dh)

    # collect
    texts = get_texts("collect")
    per_dom = (args.fit_tokens + args.test_tokens) // max(len(texts), 1)
    Kbuf = [[] for _ in range(L)]; Vbuf = [[] for _ in range(L)]
    kv.mode = "record"
    with torch.no_grad():
        for dom, text in texts.items():
            ids = tok(text, return_tensors="pt").input_ids[0][:per_dom]
            for c in range(len(ids) // args.seq_len):
                chunk = ids[c*args.seq_len:(c+1)*args.seq_len].unsqueeze(0).to(dev)
                model(chunk)
                for l in range(L):
                    Kbuf[l].append(kv.k[l][0].float().cpu())
                    Vbuf[l].append(kv.v[l][0].float().cpu())
                kv.k.clear(); kv.v.clear()
            print(f"[collect] {dom} done", flush=True)
    kv.mode = "off"

    # fit
    ntr = args.fit_tokens
    simK = np.zeros((L, Hkv, Hkv)); simV = np.zeros((L, Hkv, Hkv))
    RK, RV = [], []
    for l in range(L):
        K = torch.cat(Kbuf[l]).view(-1, Hkv, dh); Kbuf[l] = None
        V = torch.cat(Vbuf[l]).view(-1, Hkv, dh); Vbuf[l] = None
        perm = torch.randperm(K.shape[0])
        K = K[perm].to(dev); V = V[perm].to(dev)
        sK, rK = fit_layer(K[:ntr], K[ntr:])
        sV, rV = fit_layer(V[:ntr], V[ntr:])
        simK[l] = sK.numpy(); simV[l] = sV.numpy()
        RK.append(rK); RV.append(rV)
        del K, V
        if dev == "cuda": torch.cuda.empty_cache()
        print(f"[fit] layer {l+1}/{L} alignedK mean={sK.numpy()[~np.eye(Hkv,dtype=bool)].mean():.3f}", flush=True)

    off = ~np.eye(Hkv, dtype=bool)
    frac80 = float(np.mean([(simK[l][off] >= 0.80).mean() for l in range(L)]))
    frac60 = float(np.mean([(simK[l][off] >= 0.60).mean() for l in range(L)]))
    meanal = float(np.mean([simK[l][off].mean() for l in range(L)]))
    summary = [f"=== Alignment experiment ({args.model}) ===",
               f"aligned K-cosine: mean={meanal:.4f}; frac pairs >=0.80: {frac80:.4f}; >=0.60: {frac60:.4f}"]
    np.savez_compressed(os.path.join(args.out, "aligned_sim.npz"), simK=simK, simV=simV)

    # plans + simulate
    rows = ["domain,G,condition,ppl,ppl_delta_pct,top1_agree"]
    plans = {}
    for G in Gs:
        plan, naive = [], []
        for l in range(L):
            labels = cluster(simK[l], G)
            groups = {}
            for h, lab in enumerate(labels): groups.setdefault(lab, []).append(h)
            lp = []
            for members in groups.values():
                if len(members) < 2: continue
                sub = simK[l][np.ix_(members, members)]
                leader = members[int(np.argmax(sub.mean(1)))]
                for mb in members:
                    if mb == leader: continue
                    lp.append((leader, mb,
                               RK[l][leader][mb].to(dev).bfloat16(),
                               RV[l][leader][mb].to(dev).bfloat16()))
            plan.append(lp)
            naive.append(list(groups.values()))
        plans[G] = (plan, naive)

    for dom, text in get_texts("eval").items():
        ids = tok(text, return_tensors="pt").input_ids[0][:args.eval_tokens]
        if len(ids) < args.seq_len: continue
        kv.mode = "off"
        base_ppl, stored = run_pass(model, ids, args.seq_len, dev)
        rows.append(f"{dom},-,baseline,{base_ppl:.4f},0,1")
        print(f"[{dom}] baseline ppl={base_ppl:.2f}", flush=True)
        for G in Gs:
            plan, naive = plans[G]
            for name in ["aligned", "naive"]:
                if name == "aligned": kv.plan = plan; kv.mode = "share"
                else: kv.naive = naive; kv.mode = "naive"
                ppl, top1 = run_pass(model, ids, args.seq_len, dev, stored=stored)
                kv.mode = "off"
                d = 100 * (ppl - base_ppl) / base_ppl
                rows.append(f"{dom},{G},{name},{ppl:.4f},{d:.2f},{top1:.4f}")
                print(f"[{dom}] G={G} {name}: ppl={ppl:.2f} d={d:+.1f}% top1={top1:.3f}", flush=True)

    open(os.path.join(args.out, "results.csv"), "w").write("\n".join(rows))
    txt = "\n".join(summary)
    open(os.path.join(args.out, "summary.txt"), "w").write(txt)
    print(txt, flush=True)
    print("[done]", flush=True)

if __name__ == "__main__":
    main()
