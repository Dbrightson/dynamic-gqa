#!/usr/bin/env python3
"""Stage 3: Oracle simulation of KV sharing (no training, no new kernels).

For each domain, compares three conditions at equal group count G:
  baseline : unmodified model
  static   : uniform contiguous grouping of KV heads into G groups
  dynamic  : per-domain grouping from analysis clusters.json (oracle router)

Grouped heads' k_proj/v_proj outputs are overwritten with the group mean
inside the forward pass. Reports perplexity, top-1 agreement with baseline,
and mean KL(baseline || shared) of next-token distributions.

Output: results/simulation/results.csv and a printed markdown table.
"""
import argparse, json, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_eval_texts():
    from datasets import load_dataset
    out = {}
    def add(name, fn):
        try: out[name] = fn()
        except Exception as e: print(f"[data] SKIP {name}: {e}")
    # held-out splits, distinct from collection where possible
    add("prose", lambda: "\n\n".join(
        load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")["text"]))
    add("math", lambda: "\n\n".join(
        r["question"] + "\n" + r["answer"]
        for r in load_dataset("openai/gsm8k", "main", split="train").select(range(400))))
    add("code", lambda: "\n\n".join(
        r["code"] for r in load_dataset("google-research-datasets/mbpp", split="train")))
    add("stories", lambda: "\n\n".join(
        r["text"] for r in load_dataset("roneneldan/TinyStories",
                                        split="train[:2000]")))
    return out


class Sharer:
    """Hooks that overwrite grouped KV head outputs with the group mean."""
    def __init__(self, model, Hkv, dh):
        self.Hkv, self.dh = Hkv, dh
        self.groups = None            # list over layers: list of head-index lists
        self.handles = []
        for i, layer in enumerate(model.model.layers):
            self.handles.append(layer.self_attn.k_proj.register_forward_hook(self._mk(i)))
            self.handles.append(layer.self_attn.v_proj.register_forward_hook(self._mk(i)))
    def _mk(self, li):
        def hook(mod, inp, outp):
            if self.groups is None: return outp
            shp = outp.shape
            x = outp.view(*shp[:-1], self.Hkv, self.dh).clone()
            for members in self.groups[li]:
                if len(members) > 1:
                    x[..., members, :] = x[..., members, :].mean(dim=-2, keepdim=True)
            return x.view(shp)
        return hook
    def set(self, groups): self.groups = groups
    def off(self): self.groups = None
    def remove(self):
        for h in self.handles: h.remove()


def labels_to_groups(labels):
    """[0,1,0,1,...] -> [[0,2,...],[1,3,...]]"""
    g = {}
    for h, l in enumerate(labels): g.setdefault(l, []).append(h)
    return list(g.values())


@torch.no_grad()
def evaluate(model, ids, seq_len, device, base_logits=None):
    """Returns (ppl, top1_agreement, mean_kl, logits_list)."""
    nll, ntok, agree, kl_sum, kl_n = 0.0, 0, 0, 0.0, 0
    logits_out = []
    n_chunks = len(ids) // seq_len
    for c in range(n_chunks):
        chunk = ids[c*seq_len:(c+1)*seq_len].unsqueeze(0).to(device)
        out = model(chunk, labels=chunk)
        T = chunk.shape[1] - 1
        nll += out.loss.item() * T
        ntok += T
        lg = out.logits[0, :-1].float().cpu()          # [T-?, V]
        logits_out.append(lg.argmax(-1))
        if base_logits is not None:
            bl = base_logits[c]                         # stored baseline info
            agree += (lg.argmax(-1) == bl["argmax"]).sum().item()
            p = torch.log_softmax(bl["logits"], dim=-1)
            q = torch.log_softmax(lg, dim=-1)
            kl_sum += (p.exp() * (p - q)).sum(-1).mean().item() * 1
            kl_n += 1
    ppl = float(np.exp(nll / max(ntok, 1)))
    top1 = agree / max(ntok, 1) if base_logits is not None else 1.0
    kl = kl_sum / max(kl_n, 1) if base_logits is not None else 0.0
    return ppl, top1, kl


@torch.no_grad()
def baseline_pass(model, ids, seq_len, device, keep_logits_chunks=8):
    """Baseline PPL + stored logits (subsampled chunks to bound memory)."""
    nll, ntok = 0.0, 0
    stored = []
    n_chunks = len(ids) // seq_len
    keep = set(np.linspace(0, n_chunks - 1, min(keep_logits_chunks, n_chunks)).astype(int))
    for c in range(n_chunks):
        chunk = ids[c*seq_len:(c+1)*seq_len].unsqueeze(0).to(device)
        out = model(chunk, labels=chunk)
        T = chunk.shape[1] - 1
        nll += out.loss.item() * T; ntok += T
        lg = out.logits[0, :-1].float().cpu()
        stored.append({"argmax": lg.argmax(-1),
                       "logits": lg if c in keep else None})
    ppl = float(np.exp(nll / max(ntok, 1)))
    return ppl, stored, n_chunks


@torch.no_grad()
def shared_pass(model, ids, seq_len, device, stored):
    nll, ntok, agree, kl_sum, kl_n = 0.0, 0, 0, 0.0, 0
    n_chunks = len(ids) // seq_len
    for c in range(n_chunks):
        chunk = ids[c*seq_len:(c+1)*seq_len].unsqueeze(0).to(device)
        out = model(chunk, labels=chunk)
        T = chunk.shape[1] - 1
        nll += out.loss.item() * T; ntok += T
        lg = out.logits[0, :-1].float().cpu()
        agree += (lg.argmax(-1) == stored[c]["argmax"]).sum().item()
        if stored[c]["logits"] is not None:
            p = torch.log_softmax(stored[c]["logits"], -1)
            q = torch.log_softmax(lg, -1)
            kl_sum += (p.exp() * (p - q)).sum(-1).mean().item(); kl_n += 1
    return (float(np.exp(nll / max(ntok, 1))),
            agree / max(ntok, 1),
            kl_sum / max(kl_n, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--clusters", default="results/analysis/clusters.json")
    ap.add_argument("--out", default="results/simulation")
    ap.add_argument("--tokens-per-domain", type=int, default=8192)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--groups", default="2,4")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    Gs = [int(g) for g in args.groups.split(",")]

    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=args.device)
    model.eval()
    cfg = model.config
    Hq = cfg.num_attention_heads
    Hkv = getattr(cfg, "num_key_value_heads", Hq)
    dh = getattr(cfg, "head_dim", None) or cfg.hidden_size // Hq
    L = cfg.num_hidden_layers
    clus = json.load(open(args.clusters))
    sharer = Sharer(model, Hkv, dh)

    rows = ["domain,G,condition,ppl,ppl_delta_pct,top1_agree,mean_kl"]
    md = ["| domain | G | condition | PPL | ΔPPL% | top-1 agree | KL |",
          "|---|---|---|---|---|---|---|"]
    for dom, text in get_eval_texts().items():
        ids = tok(text, return_tensors="pt").input_ids[0][:args.tokens_per_domain]
        if len(ids) < args.seq_len: continue
        sharer.off()
        base_ppl, stored, _ = baseline_pass(model, ids, args.seq_len, args.device)
        rows.append(f"{dom},-,baseline,{base_ppl:.4f},0,1,0")
        md.append(f"| {dom} | – | baseline | {base_ppl:.2f} | 0 | 1.000 | 0 |")
        for G in Gs:
            # static uniform grouping, same for all layers
            static = [labels_to_groups([h * G // Hkv for h in range(Hkv)])
                      for _ in range(L)]
            # dynamic oracle grouping for this domain (fallback: first domain)
            dom_key = dom if dom in clus["labels"] else clus["domains"][0]
            dlab = clus["labels"][dom_key][str(G)]      # per-layer label lists
            dynamic = [labels_to_groups(dlab[l]) for l in range(L)]
            for name, groups in [("static", static), ("dynamic", dynamic)]:
                sharer.set(groups)
                ppl, top1, kl = shared_pass(model, ids, args.seq_len, args.device, stored)
                sharer.off()
                d = 100 * (ppl - base_ppl) / base_ppl
                rows.append(f"{dom},{G},{name},{ppl:.4f},{d:.2f},{top1:.4f},{kl:.5f}")
                md.append(f"| {dom} | {G} | {name} | {ppl:.2f} | {d:+.2f}% | {top1:.3f} | {kl:.4f} |")
                print(md[-1], flush=True)
    open(os.path.join(args.out, "results.csv"), "w").write("\n".join(rows))
    table = "\n".join(md)
    open(os.path.join(args.out, "results.md"), "w").write(table)
    print("\n=== Stage 3 results ===\n" + table)
    print("\nPre-registered criterion: dynamic PPL degradation <= 1% at a G where "
          "static degrades >= 2x as much.")

if __name__ == "__main__":
    main()
