# Dynamic Grouped-Query Attention: Training-Free KV-Cache Compression via Attention-Head Alignment

Code, data, and paper for **"Dynamic Grouped-Query Attention: Predictive, Input-Dependent Key-Value Head Sharing via Learned Collaboration Templates"** (Don Brightson, Independent Researcher, 2026).

**Total compute cost to reproduce every number in the paper: under US$40 of rented cloud GPUs** (RTX 3090/4090 and one NVIDIA A100 session). All raw result CSVs are included so the tables and figures can be verified without running anything.

## TL;DR

1. **We formalized Dynamic GQA** (input-dependent KV-head grouping) and proved its cache/FLOP savings, a cache-consistency guarantee under causal decoding, and a spectral perturbation bound on the error KV sharing introduces.
2. **The dynamic premise fails as stated.** Raw KV-head activations are nearly orthogonal (mean pairwise cosine 0.002 across Pythia-2.8B, Qwen3-4B, Nemotron-Mini-4B) and head-similarity structure barely varies across text domains (adjusted Rand index 0.59-0.86). A token-level router has little to exploit.
3. **The redundancy is real but basis-obscured.** After fitting closed-form ridge maps between head pairs (12K tokens, no training), similarity rises to 0.89-0.96 with 96-100% of pairs above 0.80.
4. **Exploiting it gives the best training-free KV compression in our controlled benchmark.** At matched cache budget, alignment-based reconstruction beats naive mean merging in all 56 comparisons (up to 3 orders of magnitude in perplexity). The multi-reference variant reaches **+17-20% perplexity at 2x compression beyond Qwen3-4B's built-in GQA with zero gradient steps**.

## Repository contents

| File | What it is |
|---|---|
| `dynamic_gqa.tex` / `dynamic_gqa_tmlr.pdf` | The paper (TMLR-format source and compiled PDF) |
| `collect_activations.py` | Stage 1: per-head K/V activation + attention-map collection (Llama-style models) |
| `analyze_redundancy.py` | Stage 2: cosine/JS similarity, clustering, cross-domain ARI, pre-registered verdicts |
| `simulate_sharing.py` | Stage 3: oracle KV-sharing simulation (static vs dynamic grouping, mean merge) |
| `mha_experiment.py` | Full Stage 1-3 pipeline for GPTNeoX models (Pythia-2.8B) |
| `align_experiment.py` | Alignment experiment for GPTNeoX: ridge maps between heads, leader+adapter sharing |
| `align_llama.py` | Alignment experiment for Llama-style models (Qwen3-4B, Nemotron-Mini-4B, Gemma) |
| `align_compare.py` | Head-to-head benchmark: naive vs Procrustes-merge vs multi-reference vs single-leader |
| `run_all.sh` | One-command runner for stages 1-3 (includes SMOKE=1 sanity mode) |
| `results_*.csv` | Raw results behind every table and figure in the paper |
| `fig_*.pdf` | The paper's figures, regenerable from the CSVs |

## Reproducing

Any 24GB+ GPU (RTX 3090/4090 class) suffices; an A100 is faster but unnecessary.

```bash
pip install "transformers>=4.51" datasets accelerate scikit-learn matplotlib
SMOKE=1 bash run_all.sh                       # ~3 min sanity check on a 135M model
bash run_all.sh Qwen/Qwen3-4B                 # stages 1-3, ~30 min on an A100
python align_llama.py --model Qwen/Qwen3-4B   # alignment experiment
python align_compare.py --model Qwen/Qwen3-4B # benchmark vs prior training-free conversions
```

Models are downloaded from Hugging Face (all ungated). Datasets: WikiText-2, GSM8K, MBPP, TinyStories.

## Key results (all training-free)

Aligned similarity of KV heads (raw cosine -> after fitted linear maps):

| Model | Raw mean cosine | Aligned mean cosine | Pairs >= 0.80 |
|---|---|---|---|
| Pythia-2.8B (MHA) | 0.002 | 0.962 | 99.75% |
| Qwen3-4B (GQA) | 0.002 | 0.893 | 95.7% |
| Nemotron-Mini-4B (GQA) | 0.002 | 0.932 | 100% |

Benchmark at matched cache budget on Qwen3-4B, 2x compression beyond built-in GQA (perplexity increase vs unmodified model):

| Method | stories | math | code | prose |
|---|---|---|---|---|
| Naive mean merge (GQA-style) | +4,639% | +5,755% | +16,537% | +7,586% |
| Procrustes align + merge (Jin et al. style) | +37% | +33% | +44% | +127% |
| **Multi-reference reconstruction (Shaikh et al. style)** | **+17%** | **+20%** | **+32%** | **+111%** |
| Single-leader + ridge adapters | +169% | +234% | +273% | +528% |

## Relation to prior work

The alignment phenomenon independently corroborates concurrent findings: Jin et al. 2024 (arXiv:2412.20677) use Procrustes alignment for MHA-to-GQA conversion, and Shaikh et al. 2026 (arXiv:2603.13314) document inter-head linear predictability at scale. Our contributions are the pre-registered falsification of *input-dependent* (token-level dynamic) grouping, the cache-consistency and perturbation theory, and the first controlled comparison of training-free conversions at strictly matched cache budget, clustering, fit data, and evaluation.

## Disclosure

This research was conducted with substantial AI assistance (Anthropic's Claude) for experiment implementation, execution, and manuscript drafting, under the direction and responsibility of the author.

## License

MIT for code; paper text CC BY 4.0.
