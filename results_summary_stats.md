# Summary statistics (Stage 2 + alignment)

## Aligned K-cosine (fitted ridge maps, held-out tokens)

- Pythia-2.8B (MHA, L=32, H=32, dh=80): mean=0.9620; frac pairs >=0.80: 0.9975; >=0.60: 1.0000
- Qwen3-4B (GQA, L=36, Hq=32, Hkv=8, dh=128): mean=0.8929; frac pairs >=0.80: 0.9573; >=0.60: 1.0000
- Nemotron-Mini-4B (GQA, L=32, Hq=24, Hkv=8, dh=128): mean=0.9322; frac pairs >=0.80: 1.0000

## Raw similarity, Qwen3-4B (per-domain means over all layers/pairs)

domain, k_cos mean, k_cos max, v_cos mean, q_js mean, q_js min
code, 0.0023, 0.3955, -0.0003, 0.2203, 0.0066
math, 0.0024, 0.4518, 0.0003, 0.2300, 0.0095
prose, 0.0024, 0.4240, -0.0001, 0.2126, 0.0069
stories, 0.0025, 0.4552, 0.0006, 0.2173, 0.0067

## Pre-registered Stage-2 verdicts (both fail => no support for token-level dynamic routing)

- Qwen3-4B: fraction of layers with rho(0.90) >= 0.30 is 0.00 for all four domains; cross-domain ARI 0.591 (G=2), 0.633 (G=4)
- Pythia-2.8B: rho(0.90) = 0.00 everywhere; cross-domain ARI 0.817 (G=4), 0.831 (G=8), 0.863 (G=16)
