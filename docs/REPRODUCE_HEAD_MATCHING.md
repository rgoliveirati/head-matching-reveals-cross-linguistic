# Reproducing the head-matching analysis

This document adds the reproducibility instructions for the revised manuscript result:
absolute-index head comparison underestimates cross-linguistic correspondence, while
permutation-aware head matching recovers head-level generalization.

## 1. Generate the aggregated attention CSV

Run the UD × attention extraction using only independent monolingual models, all target
relations, all splits, and German as the typological control:

```bash
python src/run_ud_attention_eval.py \
  --langs pt,gl,es,it,fr,ro \
  --include_control --control_langs de \
  --splits train,dev,test \
  --model_mode mono_por_lingua \
  --max_sents 1000 \
  --max_len 512 \
  --rel_filter nsubj,obj,case,amod \
  --out_csv data/outputs/attention_mono_all_splits.csv
```

Expected input columns for the next step:

```text
lang, deprel, layer, head, direction, mean_attention, n_arcs
```

The current pipeline also writes useful metadata columns such as `model_id`,
`model_family`, `treebank`, and `split`.

## 2. Compute the head-matching metrics

Place `compute_head_matching_metrics.py` under `src/` and run:

```bash
python src/compute_head_matching_metrics.py \
  --in_csv data/outputs/attention_mono_all_splits.csv \
  --direction head_to_dep \
  --split test \
  --model_family mono \
  --rels nsubj,obj,case,amod \
  --langs de,es,fr,gl,it,pt,ro \
  --layers 12 \
  --heads 12 \
  --arc_weight_agg max \
  --out_dir results
```

The script writes:

```text
results/head_matching_metrics_test_head_to_dep_mono.csv
results/pairwise_head_matching_test_head_to_dep_mono.csv
results/head_matching_permutations_test_head_to_dep_mono.csv
```

## 3. Meaning of the output columns

`head_matching_metrics_*.csv` contains one row per UD relation:

| Column | Meaning |
|---|---|
| `rho_micro_index` | Spearman correlation over 144 layer-head positions aligned by absolute index. |
| `rho_macro_mean` | Mean pairwise Spearman correlation over 12-layer profiles. |
| `rho_macro_weighted` | Same as macro, weighted by `min(n_arcs)`. |
| `rho_micro_matched_joint` | Head-matched micro correlation using all relations to estimate the matching. This is an upper bound. |
| `rho_micro_matched_loo` | Head-matched micro correlation using leave-one-out matching. This is the main reported matched value. |

`pairwise_head_matching_*.csv` contains the same metrics by language pair and relation.
This file is used to reconstruct the lower-triangular language-pair matrices.

`head_matching_permutations_*.csv` contains the actual Hungarian assignments:

```text
lang_a, lang_b, mode, deprel_eval, layer, head_a, head_b
```

For `matched_joint`, `deprel_eval = (all)` because the same matching is used for all relations.
For `matched_loo`, `deprel_eval` identifies the relation being evaluated, which was excluded
from the matching signature.

## 4. Notes on arc weights

`run_ud_attention_eval.py` stores `n_arcs` for each layer-head cell. Therefore, the same
arc count is typically repeated across the 144 positions of a language/relation profile.
For that reason, the head-matching script uses `--arc_weight_agg max` by default to recover
the per-language/per-relation support before applying the article's pair weight:

```text
w(lang_a, lang_b; relation) = min(n_arcs(lang_a, relation), n_arcs(lang_b, relation))
```

If you need exact compatibility with older scripts that summed repeated cells, use:

```bash
--arc_weight_agg sum
```

In complete 12 × 12 profiles, this only rescales all pair weights by a constant and should not
change the aggregate weighted mean.

## 5. Directional robustness

To reproduce the alternative direction check, rerun the same command with:

```bash
--direction dep_to_head
```

## 6. Recommended citation in the manuscript methods

The head matching is computed independently within each layer. The cost of matching two heads
is the Euclidean distance between their functional signatures across UD relations. The reported
LOO value excludes the relation under evaluation when estimating the matching, reducing circularity.
