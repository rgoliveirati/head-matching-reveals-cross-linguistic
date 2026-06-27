#!/usr/bin/env bash
set -euo pipefail

ATTN_CSV="${1:-data/outputs/attention_mono_all_splits.csv}"
OUT_DIR="${2:-results}"

python src/compute_head_matching_metrics.py \
  --in_csv "$ATTN_CSV" \
  --direction head_to_dep \
  --split test \
  --model_family mono \
  --rels nsubj,obj,case,amod \
  --langs de,es,fr,gl,it,pt,ro \
  --layers 12 \
  --heads 12 \
  --arc_weight_agg max \
  --out_dir "$OUT_DIR"
