#!/usr/bin/env python3
"""Stage 1+2a: Collect per-head K/V activations and attention maps from a
pretrained model, accumulate head-similarity statistics per layer per domain.

Works with Llama-style attention (q_proj/k_proj/v_proj), e.g. Qwen3-4B,
Llama-3.2-3B, SmolLM2. Inference only, no gradients.

Output (in --out dir):
  stats_<domain>.npz  with per-layer:
    k_cos  [L, Hkv, Hkv]  mean pairwise cosine similarity of key head vectors
    v_cos  [L, Hkv, Hkv]  same for value heads
    q_js   [L, Hq, Hq]    mean pairwise Jensen-Shannon divergence of
                          attention distributions between query heads
    k_sub  [L, S, Hkv, dh] subsample of raw key head vectors (for clustering)
"""
import argparse, json, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

EPS = 1e-9

def get_domain_texts(tokens_per_domain, tokenizer):
    """Return {domain: long_text} with best-effort dataset loading."""
    from datasets import load_dataset
    out = {}
    def add(name, fn):
        try:
            out[name] = fn()
            print(f"[data] loaded domain '{name}'")
        except Exception as e:
            print(f"[data] SKIP domain '{name}': {e}")
    add("prose", lambda: "\n\n".join(
        load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]))
    add("math", lambda: "\n\n".join(
        r["question"] + "\n" + r["answer"]
        for r in load_dataset("openai/gsm8k", "main", split="test")))
    add("code", lambda: "\n\n".join(
        r["code"] for r in load_dataset("google-research-datasets/mbpp", split="test")))
    add("stories", lambda: "\n\n".join(
        r["text"] for r in load_dataset("roneneldan/TinyStories", split="validation")))
    if not out:
        raise RuntimeError("No dataset could be loaded; check internet/HF access.")
    return out


class Recorder:
    """Grabs k_proj / v_proj outputs for every layer via forward hooks."""
    def __init__(self, model):
        self.k, self.v = {}, {}
        self.handles = []
        for i, layer in enumerate(model.model.layers):
            self.handles.append(layer.self_attn.k_proj.register_forward_hook(
                self._mk(self.k, i)))
            self.handles.append(layer.self_attn.v_proj.register_forward_hook(
                self._mk(self.v, i)))
    def _mk(self, store, i):
        def hook(mod, inp, outp):
            store[i] = outp.detach()
        return hook
    def clear(self):
        self.k.clear(); self.v.clear()
    def remove(self):
        for h in self.handles: h.remove()


def pairwise_cos(x):
    """x: [T, H, dh] -> [H, H] mean over T of pairwise cosine similarity."""
    xn = x / (x.norm(dim=-1, keepdim=True) + EPS)
    sim = torch.einsum("thd,tgd->thg", xn, xn)   # [T, H, H]
    return sim.mean(dim=0)


def pairwise_js(attn, n_query_samples=32):
    """attn: [Hq, T, T] causal attention rows -> [Hq, Hq] mean JS divergence."""
    Hq, T, _ = attn.shape
    lo = min(8, T - 1)
    idx = torch.linspace(lo, T - 1, steps=min(n_query_samples, T - lo)).long()
    p = attn[:, idx, :].clamp_min(EPS)                      # [Hq, S, T]
    p = p / p.sum(-1, keepdim=True)
    a = p.unsqueeze(1)                                      # [Hq, 1, S, T]
    b = p.unsqueeze(0)                                      # [1, Hq, S, T]
    m = 0.5 * (a + b)
    kl = lambda x, y: (x * (x / y).log()).sum(-1)
    js = 0.5 * kl(a, m) + 0.5 * kl(b, m)                    # [Hq, Hq, S]
    return js.mean(-1)


def offdiag_mean(m):
    L, H, _ = m.shape
    mask = ~torch.eye(H, dtype=torch.bool)
    return m[:, mask].mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--out", default="results/activations")
    ap.add_argument("--tokens-per-domain", type=int, default=8192)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--subsample-tokens", type=int, default=256,
                    help="raw K vectors kept per layer per domain")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="eager",
        device_map=args.device)
    model.eval()
    cfg = model.config
    Hq = cfg.num_attention_heads
    Hkv = getattr(cfg, "num_key_value_heads", Hq)
    dh = getattr(cfg, "head_dim", None) or cfg.hidden_size // Hq
    L = cfg.num_hidden_layers
    print(f"[model] {args.model}: L={L} Hq={Hq} Hkv={Hkv} dh={dh}")
    meta = dict(model=args.model, L=L, Hq=Hq, Hkv=Hkv, dh=dh,
                seq_len=args.seq_len, tokens_per_domain=args.tokens_per_domain)
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=2)

    rec = Recorder(model)
    domains = get_domain_texts(args.tokens_per_domain, tok)

    for dom, text in domains.items():
        ids = tok(text, return_tensors="pt").input_ids[0][:args.tokens_per_domain]
        n_chunks = len(ids) // args.seq_len
        if n_chunks == 0:
            print(f"[{dom}] too little text, skipping"); continue
        k_cos = torch.zeros(L, Hkv, Hkv)
        v_cos = torch.zeros(L, Hkv, Hkv)
        q_js = torch.zeros(L, Hq, Hq)
        k_sub = [[] for _ in range(L)]
        sub_per_chunk = max(1, args.subsample_tokens // n_chunks)
        with torch.no_grad():
            for c in range(n_chunks):
                chunk = ids[c*args.seq_len:(c+1)*args.seq_len].unsqueeze(0).to(args.device)
                rec.clear()
                outp = model(chunk, output_attentions=True)
                for l in range(L):
                    k = rec.k[l][0].float().view(-1, Hkv, dh)   # [T, Hkv, dh]
                    v = rec.v[l][0].float().view(-1, Hkv, dh)
                    k_cos[l] += pairwise_cos(k).cpu()
                    v_cos[l] += pairwise_cos(v).cpu()
                    q_js[l] += pairwise_js(outp.attentions[l][0].float()).cpu()
                    step = max(1, k.shape[0] // sub_per_chunk)
                    k_sub[l].append(k[::step][:sub_per_chunk].cpu())
                del outp
                if args.device == "cuda":
                    torch.cuda.empty_cache()
                print(f"[{dom}] chunk {c+1}/{n_chunks}", flush=True)
        k_cos /= n_chunks; v_cos /= n_chunks; q_js /= n_chunks
        k_sub = torch.stack([torch.cat(s)[:args.subsample_tokens] for s in k_sub])
        np.savez_compressed(
            os.path.join(args.out, f"stats_{dom}.npz"),
            k_cos=k_cos.numpy(), v_cos=v_cos.numpy(), q_js=q_js.numpy(),
            k_sub=k_sub.numpy().astype(np.float16))
        print(f"[{dom}] saved. mean off-diag K-cos over layers: "
              f"{offdiag_mean(k_cos):.4f}")
    rec.remove()
    print("[done] Stage 1 complete.")


if __name__ == "__main__":
    main()
