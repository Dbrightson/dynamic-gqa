#!/usr/bin/env bash
# One-command runner for the Dynamic GQA validation experiments.
# Usage:  bash run_all.sh [MODEL]
# Default model: Qwen/Qwen3-4B (ungated). For Llama-3.2-3B you must first:
#   export HF_TOKEN=hf_...   (with license accepted on huggingface.co)
set -euo pipefail
MODEL="${1:-Qwen/Qwen3-4B}"
SMOKE="${SMOKE:-0}"   # SMOKE=1 bash run_all.sh -> tiny model, quick sanity run

if [ "$SMOKE" = "1" ]; then
  MODEL="HuggingFaceTB/SmolLM2-135M"
  EXTRA="--tokens-per-domain 1024 --seq-len 256"
else
  EXTRA=""
fi

echo "=== [0/3] deps ==="
pip install -q --upgrade "transformers>=4.51" datasets accelerate scikit-learn matplotlib sentencepiece

echo "=== [1/3] Stage 1: collect activations ($MODEL) ==="
python collect_activations.py --model "$MODEL" --out results/activations $EXTRA

echo "=== [2/3] Stage 2: redundancy analysis ==="
python analyze_redundancy.py --acts results/activations --out results/analysis

echo "=== [3/3] Stage 3: KV-sharing simulation ==="
python simulate_sharing.py --model "$MODEL" --clusters results/analysis/clusters.json \
  --out results/simulation $EXTRA

tar czf results.tar.gz results/
echo ""
echo "ALL DONE. Key outputs:"
echo "  results/analysis/summary.txt   <- Stage 2 verdicts"
echo "  results/simulation/results.md  <- Stage 3 table"
echo "  results.tar.gz                 <- everything, download this"
cat results/analysis/summary.txt
