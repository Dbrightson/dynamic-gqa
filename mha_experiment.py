#!/usr/bin/env python3
"""MHA experiment: full Stage 1-3 pipeline for GPTNeoX models (fused QKV),
e.g. EleutherAI/pythia-2.8b (32 heads, true multi-head attention).

Measures per-head K/V redundancy across domains, clusters heads, then
simulates static vs dynamic-oracle KV-head sharing. One file, one run.
"""
import argparse, json, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

EPS = 1e-9

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
    """Hooks on query_key_value: record or overwrite per-head K/V slices.
    GPTNeoX layout: qkv.view(B,T,H,3*dh); q=[...,:dh], k=[...,dh:2dh], v=[...,2dh:]."""
    def __init__(self, model, H, dh):
        self.H, self.dh = H, dh
        self.mode = "off"          # off | record | share
        self.k, self.v = {}, {}
        self.groups = None
        self.handles = []
        for i, layer in enumerate(model.gpt_neox.layers):
            self.handles.append(layer.attention.query_key_value.register_forward_hook(self._mk(i)))
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
            for members in self.groups[li]:
                if len(members) > 1:
                    x[..., members, self.dh:2*self.dh] = x[..., members, self.dh:2*self.dh].mean(dim=-2, keepdim=True)
                    x[..., members, 2*self.dh:] = x[..., members, 2*self.dh:].mean(dim=-2, keepdim=True)
            return x.view(shp)
        return hook


def pairwise_cos(x):
    xn = x / (x.norm(dim=-1, keepdim=True) + EPS)
    return torch.einsum("thd,tgd->thg", xn, xn).mean(dim=0)


def cluster(sim, G):
    from sklearn.cluster import AgglomerativeClustering
    G = min(G, sim.shape[0])
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    dist = np.clip((dist + dist.T) / 2, 0, None)
    return AgglomerativeClustering(n_clusters=G, metric="precomputed",
                                   linkage="average").fit_predict(dist).tolist()


def labels_to_groups(labels):
    g = {}
    for h, l in enumerate(labels): g.setdefault(l, []).append(h)
    return list(g.values())


@torch.no_grad()
def run_pass(model, ids, seq_len, device, stored=None, keep=8):
    nll, ntok, agree, kl_sum, kl_n = 0.0, 0, 0, 0.0, 0
    new_store = []
    n_chunks = len(ids) // seq_len
    keep_set = set(np.linspace(0, n_chunks-1, min(keep, n_chunks)).astype(int))
    for c in range(n_chunks):
        chunk = ids[c*seq_len:(c+1)*seq_len].unsqueeze(0).to(device)
        out = model(chunk, labels=chunk)
        T = chunk.shape[1] - 1
        nll += out.loss.item() * T; ntok += T
        lg = out.logits[0, :-1].float().cpu()
        if stored is None:
            new_store.append({"argmax": lg.argmax(-1), "logits": lg if c in keep_set else None})
        else:
            agree += (lg.argmax(-1) == stored[c]["argmax"]).sum().item()
            if stored[c]["logits"] is not None:
                p = torch.log_softmax(stored[c]["logits"], -1)
                q = torch.log_softmax(lg, -1)
                kl_sum += (p.exp() * (p - q)).sum(-1).mean().item(); kl_n += 1
    ppl = float(np.exp(nll / max(ntok, 1)))
    if stored is None: return ppl, new_store
    return ppl, agree / max(ntok, 1), kl_sum / max(kl_n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-2.8b")
    ap.add_argument("--out", default="results_mha")
    ap.add_argument("--tokens-per-domain", type=int, default=6144)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--groups", default="4,8,16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    Gs = [int(g) for g in args.groups.split(",")]

    from sklearn.metrics import adjusted_rand_score
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype,
                                                 device_map=args.device)
    model.eval()
    cfg = model.config
    H = cfg.num_attention_heads
    dh = cfg.hidden_size // H
    L = cfg.num_hidden_layers
    print(f"[model] {args.model}: L={L} H={H} dh={dh}", flush=True)
    kv = NeoXKV(model, H, dh)

    # ---- Stage 1: collect K/V similarity stats -------------------------
    k_cos = {}   # dom -> [L,H,H]
    for dom, text in get_texts("collect").items():
        ids = tok(text, return_tensors="pt").input_ids[0][:args.tokens_per_domain]
        n_chunks = len(ids) // args.seq_len
        if n_chunks == 0: continue
        acc = torch.zeros(L, H, H)
        kv.mode = "record"
        with torch.no_grad():
            for c in range(n_chunks):
                chunk = ids[c*args.seq_len:(c+1)*args.seq_len].unsqueeze(0).to(args.device)
                model(chunk)
                for l in range(L):
                    acc[l] += pairwise_cos(kv.k[l][0].float()).cpu()
                kv.k.clear(); kv.v.clear()
                print(f"[{dom}] chunk {c+1}/{n_chunks}", flush=True)
        kv.mode = "off"
        k_cos[dom] = (acc / n_chunks).numpy()

    doms = list(k_cos)
    off = ~np.eye(H, dtype=bool)
    # redundancy profile
    summary = ["=== MHA (Pythia) Stage 2 verdicts ==="]
    red = {}
    for dom in doms:
        frac_layers = float(np.mean([ (k_cos[dom][l][off] > 0.90).mean() >= 0.30 for l in range(L) ]))
        mean_rho = float(np.mean([ (k_cos[dom][l][off] > 0.90).mean() for l in range(L) ]))
        red[dom] = dict(frac_layers=frac_layers, mean_rho090=mean_rho)
        summary.append(f"[{dom}] mean rho(0.90)={mean_rho:.3f}; frac layers passing >=0.30: {frac_layers:.2f} ({'PASS' if frac_layers >= 0.25 else 'fail'})")
    # clustering + ARI
    labels = {dom: {G: [cluster(k_cos[dom][l], G) for l in range(L)] for G in Gs} for dom in doms}
    for G in Gs:
        aris = [adjusted_rand_score(labels[a][G][l], labels[b][G][l])
                for l in range(L) for i, a in enumerate(doms) for b in doms[i+1:]]
        m = float(np.mean(aris)) if aris else float("nan")
        summary.append(f"[G={G}] mean cross-domain ARI: {m:.3f} ({'PASS (input-dependent)' if m < 0.5 else 'fail (static)'})")
    np.savez_compressed(os.path.join(args.out, "k_cos.npz"), **{d: k_cos[d] for d in doms})
    json.dump({"domains": doms, "groups": Gs, "labels": labels, "redundancy": red},
              open(os.path.join(args.out, "clusters.json"), "w"))

    # ---- Stage 3: simulation -------------------------------------------
    rows = ["domain,G,condition,ppl,ppl_delta_pct,top1_agree,mean_kl"]
    for dom, text in get_texts("eval").items():
        ids = tok(text, return_tensors="pt").input_ids[0][:args.tokens_per_domain]
        if len(ids) < args.seq_len: continue
        kv.mode = "off"
        base_ppl, stored = run_pass(model, ids, args.seq_len, args.device)
        rows.append(f"{dom},-,baseline,{base_ppl:.4f},0,1,0")
        print(f"[{dom}] baseline ppl={base_ppl:.2f}", flush=True)
        for G in Gs:
            static = [labels_to_groups([h * G // H for h in range(H)]) for _ in range(L)]
            dom_key = dom if dom in labels else doms[0]
            dynamic = [labels_to_groups(labels[dom_key][G][l]) for l in range(L)]
            for name, groups in [("static", static), ("dynamic", dynamic)]:
                kv.groups = groups; kv.mode = "share"
                ppl, top1, klv = run_pass(model, ids, args.seq_len, args.device, stored=stored)
                kv.mode = "off"
                d = 100 * (ppl - base_ppl) / base_ppl
                rows.append(f"{dom},{G},{name},{ppl:.4f},{d:.2f},{top1:.4f},{klv:.5f}")
                print(f"[{dom}] G={G} {name}: ppl={ppl:.2f} d={d:+.1f}% top1={top1:.3f} kl={klv:.4f}", flush=True)
    open(os.path.join(args.out, "results.csv"), "w").write("\n".join(rows))
    txt = "\n".join(summary)
    open(os.path.join(args.out, "summary.txt"), "w").write(txt)
    print(txt, flush=True)
    print("[done] MHA experiment complete.", flush=True)

if __name__ == "__main__":
    main()
