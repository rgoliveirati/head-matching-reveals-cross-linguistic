#!/usr/bin/env bash
set -euo pipefail

python src/compute_control_analyses.py \
  --attention_csv data/attention_all_splits.csv \
  --out_dir results/controls \
  --rels nsubj,obj,case,amod \
  --langs de,es,fr,gl,it,pt,ro \
  --direction head_to_dep \
  --split test \
  --model_family mono \
  --layers 12 \
  --heads 12 \
  --arc_weight_agg max \
  --gl_model_a_csv data/outputs/gl_marcosgg.csv \
  --gl_model_b_csv data/outputs/gl_fpuentes.csv \
  --gl_model_a_label gl_marcosgg \
  --gl_model_b_label gl_fpuentes
