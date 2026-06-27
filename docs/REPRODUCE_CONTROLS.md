# Reproducing Control Analyses

This document explains how to reproduce the control analyses reported in the manuscript.

## Main command

Run from the repository root:

```bash
bash scripts/reproduce_controls.sh
```

Equivalent explicit command:

```bash
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
```

## Expected outputs

The script writes the following files into `results/controls/`:

```text
main_head_matching_summary.csv
main_pairwise_head_matching.csv
without_french_control.csv
train_test_stability.csv
direction_control.csv
mono_vs_mbert_control.csv
without_pt_gl_pair.csv
romance_vs_german_control.csv
galician_two_models_control.csv
controls_manifest.csv
```

## Controls covered

| Output file | Purpose |
|---|---|
| `without_french_control.csv` | Tests whether the French CamemBERT/RoBERTa model drives the aggregate result. |
| `train_test_stability.csv` | Compares train and test aggregate metrics. |
| `direction_control.csv` | Compares `head_to_dep` and `dep_to_head`. |
| `mono_vs_mbert_control.csv` | Compares independent monolingual models with mBERT profiles. |
| `without_pt_gl_pair.csv` | Reports pairwise aggregate metrics with and without the Portuguese--Galician pair. |
| `romance_vs_german_control.csv` | Compares Romance--Romance pairs, excluding Portuguese--Galician, with German--Romance pairs. |
| `galician_two_models_control.csv` | Compares two independently trained Galician BERT models. |

## Notes

The controls use the same matching protocol as the main analysis:

- index-based micro correlation;
- macro layer-level correlation;
- matched joint correlation;
- matched leave-one-out correlation.

The preferred non-circular estimate remains `rho_micro_matched_loo`.

The two-Galician-model control requires the auxiliary files:

```text
data/outputs/gl_marcosgg.csv
data/outputs/gl_fpuentes.csv
```

If those files are absent, the script still runs but marks the Galician control as not computed.
