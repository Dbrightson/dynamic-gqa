#!/usr/bin/env python3
"""Alignment experiment (reverse-engineering head similarity).

Hypothesis: raw cosine between different heads' K/V activations is the wrong
metric because each head writes in its own basis. Test whether heads are
redundant UP TO A LINEAR MAP: fit R_{g->h} minimizing ||K_g R - K_h|| on
train tokens, measure ALIGNED COSINE on held-out tokens, and simulate
sharing where only a group leader's K/V is kept and members are
reconstructed via their fitted adapters (k_h := k_leader @ R, v_h := v_leader @ M).

GPTNeoX models (e.g. EleutherAI/pythia-2.8b). Outputs to results_align/.
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


class NeoXKV:
    """record: stash per-head pre-RoPE K,V.  share: rebuild member heads
    from their group leader via fitted linear adapters."""
    def __init__(self, model, H, dh):
        self.H, self.dh = H, dh
        self.mode = "off"
        self.k, self.v = {}, {}
        self.plan = None   # per layer: list of (leader, member, Rk [dh,dh], Rv) on device
        self.naive = None  # per layer: list of member lists (mean merge, for contrast)
        for i, layer in enumerate(model.gpt_neox.layers):
            layer.attention.query_key_value.register_forward_hook(self._mk(i))
    def _mk(self, li):
        def hook(mod, inp, outp):
            if self.mode == "off": return outp
            shp = outp.shape
            x = outp.view(*shp[:-1], self.H, 3 * self.dh)
            if self.mode == "record":
                self.k[li] = x[..., self.dh:2*self.dh].detach()
                self.v[li] = x[..., 2*self.dh:].detach()
                return outp
            x = x.clone()
            if self.mode == "share":
                for (ld, mb, Rk, Rv) in self.plan[li]:
                    x[..., mb, self.dh:2*self.dh] = x[..., ld, self.dh:2*self.dh] @ Rk
                    x[..., mb, 2*self.dh:] = x[..., ld, 2*self.dh:] @ Rv
            elif self.mode == "naive":
                for members in self.naive[li]:
                    if len(members) > 1:
                        x[..., members, self.dh:2*self.dh] = x[..., members, self.dh:2*self.dh].mean(-2, keepdim=True)
                        x[..., members, 2*self.dh:] = x[..., members, 2*self.dh:].mean(-2, keepdim=True)
            return x.view(shp)
        return hook


def fit_layer(Xtr, Xte, lam=1e-3):
    """Xtr: [Ntr,H,dh] train activations for one layer (one of K or V).
    Returns aligned_cos [H,H] on test set, and a fitter for specific pairs."""
    N, H, dh = Xtr.shape
    dev = Xtr.device
    # cross-covariances C[g,h] = X_g^T X_h  -> [H,H,dh,dh]
    C = torch.einsum("nga,nhb->ghab", Xtr, Xtr) / N
    eye = torch.eye(dh, device=dev)
    sim = torch.zeros(H, H)
    R_all_diag = {}
    for g in range(H):
        A = C[g, g] + lam * eye                      # [dh,dh]
        R = torch.linalg.solve(A, C[g])              # [H,dh,dh], R[h] maps g->h
        pred = torch.einsum("na,hab->nhb", Xte[:, g, :], R)   # [Nte,H,dh]
        pn = pred / (pred.norm(dim=-1, keepdim=True) + 1e-9)
        tn = Xte / (Xte.norm(dim=-1, keepdim=True) + 1e-9)
        sim[g] = (pn * tn).sum(-1).mean(0).cpu()     # mean cosine(pred, true)
        R_all_diag[g] = R.cpu()                      # keep for plan building
    return sim, R_all_diag


def cluster(sim, G):
    from sklearn.cluster import AgglomerativeClustering
    G = min(G, sim.shape[0])
    s = np.minimum(sim, sim.T)                       # symmetric: worst direction
    dist = np.clip(1.0 - s, 0, None)
    np.fill_diagonal(dist, 0.0)
    return AgglomerativeClustering(n_clusters=G, metric="precomputed",
                                   linkage="average").fit_predict(dist).tolist()


@torch.no_grad()
def run_pass(model, ids, seq_len, device, stored=None, keep=6):
    nll, ntok, agree = 0.0, 0, 0
    new_store = []
    n_chunks = len(ids) // seq_len
    for c in range(n_chunks):
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
    ap.add_argument("--model", default="EleutherAI/pythia-2.8b")
    ap.add_argument("--out", default="results_align")
    ap.add_argument("--fit-tokens", type=int, default=12288)
    ap.add_argument("--test-tokens", type=int, default=4096)
    ap.add_argument("--eval-tokens", type=int, default=4096)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--groups", default="8,16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    Gs = [int(g) for g in args.groups.split(",")]
    dev = args.device

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16 if dev == "cuda" else torch.float32,
        device_map=dev)
    model.eval()
    cfg = model.config
    H = cfg.num_attention_heads
    dh = cfg.hidden_size // H
    L = cfg.num_hidden_layers
    print(f"[model] {args.model}: L={L} H={H} dh={dh}", flush=True)
    kv = NeoXKV(model, H, dh)

    # ---- collect pooled activations (train+test split) ------------------
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

    # ---- fit alignments per layer ---------------------------------------
    ntr = args.fit_tokens
    simK = np.zeros((L, H, H)); simV = np.zeros((L, H, H))
    RK, RV = [], []
    for l in range(L):
        K = torch.cat(Kbuf[l]).view(-1, H, dh); Kbuf[l] = None
        V = torch.cat(Vbuf[l]).view(-1, H, dh); Vbuf[l] = None
        perm = torch.randperm(K.shape[0])
        K = K[perm].to(dev); V = V[perm].to(dev)
        sK, rK = fit_layer(K[:ntr], K[ntr:])
        sV, rV = fit_layer(V[:ntr], V[ntr:])
        simK[l] = sK.numpy(); simV[l] = sV.numpy()
        RK.append(rK); RV.append(rV)
        del K, V
        if dev == "cuda": torch.cuda.empty_cache()
        print(f"[fit] layer {l+1}/{L} alignedK mean={sK.numpy()[~np.eye(H,dtype=bool)].mean():.3f}", flush=True)

    off = ~np.eye(H, dtype=bool)
    frac80 = float(np.mean([ (simK[l][off] >= 0.80).mean() for l in range(L) ]))
    frac60 = float(np.mean([ (simK[l][off] >= 0.60).mean() for l in range(L) ]))
    meanal = float(np.mean([ simK[l][off].mean() for l in range(L) ]))
    summary = ["=== Alignment experiment (Pythia-2.8b) ===",
               f"aligned K-cosine: mean={meanal:.4f}; frac pairs >=0.80: {frac80:.4f}; >=0.60: {frac60:.4f}",
               f"(raw cosine baseline from prior run: mean 0.002, frac >=0.80: 0.0000)"]
    np.savez_compressed(os.path.join(args.out, "aligned_sim.npz"), simK=simK, simV=simV)

    # ---- build sharing plans and simulate --------------------------------
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
                               RK[l][leader][mb].to(dev).half(),
                               RV[l][leader][mb].to(dev).half()))
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
                if name == "aligned":
                    kv.plan = plan; kv.mode = "share"
                else:
                    kv.naive = naive; kv.mode = "naive"
                ppl, top1 = run_pass(model, ids, args.seq_len, dev, stored=stored)
                kv.mode = "off"
                d = 100 * (ppl - base_ppl) / base_ppl
                rows.append(f"{dom},{G},{name},{ppl:.4f},{d:.2f},{top1:.4f}")
                print(f"[{dom}] G={G} {name}: ppl={ppl:.2f} d={d:+.1f}% top1={top1:.3f}", flush=True)

    open(os.path.join(args.out, "results.csv"), "w").write("\n".join(rows))
    txt = "\n".join(summary)
    open(os.path.join(args.out, "summary.txt"), "w").write(txt)
    print(txt, flush=True)
    print("[done] alignment experiment complete.", flush=True)

if __name__ == "__main__":
    main()
