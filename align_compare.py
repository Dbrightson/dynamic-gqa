#!/usr/bin/env python3
"""Head-to-head comparison of training-free KV-sharing conversions at
matched cache budget (G stored heads per layer), Llama-style models.

Conditions:
  naive      : cluster + mean merge (GQA-style conversion baseline)
  procrustes : orthogonal Procrustes alignment to leader basis, merge the
               aligned heads, map back per member (Jin et al. 2024 style,
               training-free reimplementation)
  multiref   : reconstruct each eliminated head from ALL G stored leaders
               via ridge maps (Shaikh et al. 2026 style, matched budget)
  ours       : leader + per-pair ridge adapter (single reference)

All fitted on the same train tokens, same clustering, same eval.
"""
import argparse, os
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


class KVHooks:
    """Modes: off | record | naive | ours | procrustes | multiref."""
    def __init__(self, layers, Hkv, dh):
        self.Hkv, self.dh = Hkv, dh
        self.mode = "off"
        self.k, self.v = {}, {}
        self.cfg = None    # per-layer plan for the active condition
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
            plan = self.cfg[li]
            if self.mode == "naive":
                for members in plan:
                    if len(members) > 1:
                        x[..., members, :] = x[..., members, :].mean(-2, keepdim=True)
            elif self.mode == "ours":
                for (ld, mb, Rk, Rv) in plan:
                    x[..., mb, :] = x[..., ld, :] @ (Rk if which == "k" else Rv)
            elif self.mode == "procrustes":
                # plan: list of (members, {m: (RkOrth, RvOrth)})  leader incl. w/ identity
                for members, rots in plan:
                    if len(members) < 2: continue
                    R = {m: rots[m][0 if which == "k" else 1] for m in members}
                    shared = torch.stack([x[..., m, :] @ R[m] for m in members], dim=-2).mean(-2)
                    for m in members:
                        x[..., m, :] = shared @ R[m].transpose(-1, -2)
            elif self.mode == "multiref":
                leaders, recs = plan   # leaders: list of head idx; recs: {h: W [G*dh, dh]}
                base = torch.cat([x[..., l, :] for l in leaders], dim=-1)  # [..., G*dh]
                for h, W in recs.items():
                    x[..., h, :] = base @ (W[0] if which == "k" else W[1])
            return x.view(shp)
        return hook


def ridge(X, Y, lam=1e-3):
    """X:[N,a] Y:[N,b] -> W:[a,b]"""
    A = X.T @ X / X.shape[0] + lam * torch.eye(X.shape[1], device=X.device)
    return torch.linalg.solve(A, X.T @ Y / X.shape[0])


def procrustes(Xm, Xl):
    """orthogonal R minimizing ||Xm R - Xl||: R = U V^T, U S V^T = svd(Xm^T Xl)"""
    M = Xm.T @ Xl
    U, S, Vt = torch.linalg.svd(M)
    return U @ Vt


def cluster(sim, G):
    from sklearn.cluster import AgglomerativeClustering
    G = min(G, sim.shape[0])
    s = np.minimum(sim, sim.T)
    dist = np.clip(1.0 - s, 0, None)
    np.fill_diagonal(dist, 0.0)
    return AgglomerativeClustering(n_clusters=G, metric="precomputed",
                                   linkage="average").fit_predict(dist).tolist()


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
    ap.add_argument("--out", default="results_compare")
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
    layers = model.model.layers
    L = len(layers)
    print(f"[model] {args.model}: L={L} Hq={Hq} Hkv={Hkv} dh={dh}", flush=True)
    kv = KVHooks(layers, Hkv, dh)

    # ---- collect ----
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

    # ---- fit everything per layer ----
    ntr = args.fit_tokens
    Ktr, Kte, Vtr = [], [], []
    simK = np.zeros((L, Hkv, Hkv))
    for l in range(L):
        K = torch.cat(Kbuf[l]).view(-1, Hkv, dh); Kbuf[l] = None
        V = torch.cat(Vbuf[l]).view(-1, Hkv, dh); Vbuf[l] = None
        perm = torch.randperm(K.shape[0])
        K = K[perm].to(dev); V = V[perm].to(dev)
        Ktr.append(K[:ntr]); Kte.append(K[ntr:]); Vtr.append(V[:ntr])
        # aligned similarity for clustering (leader-style ridge, K only)
        for g in range(Hkv):
            R = ridge(K[:ntr, g, :], K[:ntr].reshape(ntr, -1)).view(dh, Hkv, dh)
            pred = torch.einsum("na,hab->nhb", K[ntr:, g, :], R.permute(1, 0, 2))
            pn = pred / (pred.norm(dim=-1, keepdim=True) + 1e-9)
            tn = Kte[l] / (Kte[l].norm(dim=-1, keepdim=True) + 1e-9)
            simK[l, g] = (pn * tn).sum(-1).mean(0).cpu().numpy()
        print(f"[sim] layer {l+1}/{L}", flush=True)

    plans = {}
    for G in Gs:
        naive_p, ours_p, proc_p, multi_p = [], [], [], []
        for l in range(L):
            labels = cluster(simK[l], G)
            groups = {}
            for h, lab in enumerate(labels): groups.setdefault(lab, []).append(h)
            glist = list(groups.values())
            naive_p.append(glist)
            # leaders
            leaders = []
            ours_l, proc_l = [], []
            for members in glist:
                sub = simK[l][np.ix_(members, members)]
                leader = members[int(np.argmax(sub.mean(1)))]
                leaders.append(leader)
                if len(members) < 2:
                    proc_l.append((members, {members[0]: (torch.eye(dh, device=dev).bfloat16(),)*2}))
                    continue
                rots = {}
                for m in members:
                    if m == leader:
                        rots[m] = (torch.eye(dh, device=dev).bfloat16(),)*2
                    else:
                        Rk = ridge(Ktr[l][:, m, :], torch.empty(0)) if False else None
                        rk = procrustes(Ktr[l][:, m, :], Ktr[l][:, leader, :]).bfloat16()
                        rv = procrustes(Vtr[l][:, m, :], Vtr[l][:, leader, :]).bfloat16()
                        rots[m] = (rk, rv)
                        ours_l.append((leader, m,
                                       ridge(Ktr[l][:, leader, :], Ktr[l][:, m, :]).bfloat16(),
                                       ridge(Vtr[l][:, leader, :], Vtr[l][:, m, :]).bfloat16()))
                proc_l.append((members, rots))
            # multiref: reconstruct every non-leader from concat of leaders
            Xk = torch.cat([Ktr[l][:, ld, :] for ld in leaders], dim=-1)
            Xv = torch.cat([Vtr[l][:, ld, :] for ld in leaders], dim=-1)
            recs = {}
            for h in range(Hkv):
                if h in leaders: continue
                recs[h] = (ridge(Xk, Ktr[l][:, h, :]).bfloat16(),
                           ridge(Xv, Vtr[l][:, h, :]).bfloat16())
            multi_p.append((leaders, recs))
            ours_p.append(ours_l)
            proc_p.append(proc_l)
        plans[G] = {"naive": naive_p, "ours": ours_p, "procrustes": proc_p, "multiref": multi_p}
        print(f"[plans] G={G} built", flush=True)
    del Ktr, Kte, Vtr
    if dev == "cuda": torch.cuda.empty_cache()

    # ---- simulate ----
    rows = ["domain,G,condition,ppl,ppl_delta_pct,top1_agree"]
    for dom, text in get_texts("eval").items():
        ids = tok(text, return_tensors="pt").input_ids[0][:args.eval_tokens]
        if len(ids) < args.seq_len: continue
        kv.mode = "off"
        base_ppl, stored = run_pass(model, ids, args.seq_len, dev)
        rows.append(f"{dom},-,baseline,{base_ppl:.4f},0,1")
        print(f"[{dom}] baseline ppl={base_ppl:.2f}", flush=True)
        for G in Gs:
            for name in ["naive", "procrustes", "multiref", "ours"]:
                kv.cfg = plans[G][name]; kv.mode = name
                ppl, top1 = run_pass(model, ids, args.seq_len, dev, stored=stored)
                kv.mode = "off"
                d = 100 * (ppl - base_ppl) / base_ppl
                rows.append(f"{dom},{G},{name},{ppl:.4f},{d:.2f},{top1:.4f}")
                print(f"[{dom}] G={G} {name}: ppl={ppl:.2f} d={d:+.1f}% top1={top1:.3f}", flush=True)

    open(os.path.join(args.out, "results.csv"), "w").write("\n".join(rows))
    print("[done]", flush=True)

if __name__ == "__main__":
    main()
