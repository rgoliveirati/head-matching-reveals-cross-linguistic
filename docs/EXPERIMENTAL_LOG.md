# Experimental Log

Project: **Head Matching Reveals Cross-Linguistic Generalization of Syntactic Attention in Monolingual Transformers**

This document records the experimental configuration used to reproduce the main results reported in the manuscript. It is intended to accompany the repository and to make explicit which information is recorded, which outputs are available, and which information was not recorded in the current experimental log.

---

## 1. Study design

The study evaluates whether syntactic attention profiles associated with Universal Dependencies relations are comparable across independently trained monolingual Transformer models.

The central methodological question is whether comparing attention heads by absolute index underestimates cross-model similarity. To address this, the repository provides both:

1. an index-based comparison of attention heads;
2. a permutation-aware head-matching comparison using optimal assignment with the Hungarian algorithm.

The analysis is observational. It identifies recoverable functional correspondence among attention heads, but it does not establish that the matched heads are causally necessary for syntactic behavior.

---

## 2. Languages and model family

The main analysis uses independent monolingual models for seven languages:

| Language | Code | Role |
|---|---:|---|
| Portuguese | `pt` | Romance language |
| Galician | `gl` | Romance language |
| Spanish | `es` | Romance language |
| French | `fr` | Romance language; RoBERTa-family model |
| Italian | `it` | Romance language |
| Romanian | `ro` | Romance language |
| German | `de` | typologically distinct control |

The main reported results use:

```text
model_family = mono
```

The pipeline also supports `mbert`, `mono_vs_mbert`, and manual model comparisons, but the main manuscript tables are based on the monolingual setting unless explicitly stated otherwise.

---

## 3. Hugging Face models

| Language | Model ID | Notes |
|---|---|---|
| Portuguese | `neuralmind/bert-base-portuguese-cased` | BERT-base architecture |
| Galician | `marcosgg/bert-base-gl-cased` | BERT-base architecture |
| Spanish | `dccuchile/bert-base-spanish-wwm-cased` | BERT-base architecture |
| French | `camembert-base` | RoBERTa-family model; treated as a controlled architectural variable |
| Italian | `dbmdz/bert-base-italian-xxl-cased` | BERT-base architecture |
| Romanian | `dumitrescustefan/bert-base-romanian-cased-v1` | BERT-base architecture |
| German | `bert-base-german-cased` | BERT-base architecture |

Additional models configured for control analyses:

| Language / condition | Model ID |
|---|---|
| Galician alternative model | `fpuentes/bert-galician` |
| Portuguese large model | `neuralmind/bert-large-portuguese-cased` |
| multilingual baseline | `bert-base-multilingual-cased` |
| XLM-R baseline | `xlm-roberta-base` |

Tokenizer versions and exact model revisions were **not recorded in the current experimental log**.

---

## 4. Universal Dependencies treebanks

| Language | Treebank repository | Prefix |
|---|---|---|
| Portuguese | `UD_Portuguese-Bosque` | `pt_bosque-ud` |
| Galician | `UD_Galician-TreeGal` | `gl_treegal-ud` |
| Spanish | `UD_Spanish-AnCora` | `es_ancora-ud` |
| French | `UD_French-GSD` | `fr_gsd-ud` |
| Italian | `UD_Italian-ISDT` | `it_isdt-ud` |
| Romanian | `UD_Romanian-RRT` | `ro_rrt-ud` |
| German | `UD_German-GSD` | `de_gsd-ud` |

The current repository downloader retrieves treebanks from the Universal Dependencies GitHub repositories using the `main.zip` or `master.zip` branch archive.

The exact UD release version and commit hash were **not recorded in the current experimental log**. For stricter reproducibility, future runs should pin each treebank to a UD release or commit hash.

---

## 5. Data splits and fallback rule

The main pipeline supports the following splits:

```text
train, dev, test
```

The final manuscript results use the `test` split for the head-matching analysis.

Galician TreeGal does not provide a `dev` split in the current pipeline configuration. When `dev` is requested for Galician, the pipeline records the split as:

```text
train_fallback_dev
```

This fallback is explicitly considered in the manuscript limitations because Galician is the smallest treebank in the analyzed set and the Portuguese--Galician pair is an exceptional case.

---

## 6. Syntactic relations

The study analyzes four Universal Dependencies relations:

| Relation | Linguistic role in the study |
|---|---|
| `nsubj` | nominal subject; central argument-structure relation |
| `obj` | object; central argument-structure relation |
| `case` | functional/grammatical marking relation |
| `amod` | adjectival modifier; nominal modification relation |

The default relation filter is:

```text
nsubj,obj,amod,case
```

The order in the paper tables may be reported as:

```text
nsubj,obj,case,amod
```

---

## 7. Attention extraction configuration

The extraction pipeline computes attention for all layers and heads of each model.

Default configuration used for the manuscript:

| Parameter | Value |
|---|---:|
| Maximum sentences per split | `1000` |
| Maximum subword length | `512` |
| Layers | `12` |
| Heads per layer | `12` |
| Total layer-head positions | `144` |
| Main direction | `head_to_dep` |
| Alternative direction | `dep_to_head` |
| Main model family | `mono` |

Attention between UD tokens is computed by aligning UD word spans to tokenizer offsets and averaging attention over all subword pairs in the head-token × dependent-token block.

Sentences for which token-subword alignment or model inference failed were skipped by the extraction script. Counts of skipped sentences, alignment failures, and inference failures were **not recorded in the current experimental log**.

---

## 8. Main input file

The repository version evaluated for the current manuscript includes the aggregated attention file:

```text
data/attention_all_splits.csv
```

Expected columns include:

```text
lang
lang_group
model_id
model_family
treebank
split
deprel
layer
head
direction
mean_attention
std_attention
n_arcs
```

The evaluated file contains monolingual and mBERT profiles, both attention directions, and train/dev/test-related splits. The main paper results are obtained by filtering:

```text
model_family = mono
direction = head_to_dep
split = test
relations = nsubj,obj,case,amod
languages = de,es,fr,gl,it,pt,ro
```

---

## 9. Reproducing attention extraction

To regenerate the aggregated attention file from the UD treebanks and Hugging Face models, run:

```bash
python src/run_ud_attention_eval.py \
  --langs pt,gl,es,it,fr,ro \
  --include_control \
  --control_langs de \
  --splits train,dev,test \
  --model_mode mono_por_lingua \
  --max_sents 1000 \
  --max_len 512 \
  --rel_filter nsubj,obj,amod,case \
  --out_csv data/attention_all_splits.csv
```

The pipeline automatically downloads the UD treebanks if they are not already present locally.

---

## 10. Reproducing index-based micro/macro metrics and entropy

To reproduce the index-based micro/macro metrics and entropy tables, run:

```bash
python src/compute_generalization_metrics.py \
  --in_csv data/attention_all_splits.csv \
  --direction head_to_dep \
  --splits test \
  --rels nsubj,obj,case,amod \
  --model_family mono \
  --out_dir results/generalization_metrics
```

Expected outputs:

```text
results/generalization_metrics/generalization_metrics_by_deprel.csv
results/generalization_metrics/entropy_by_lang.csv
```

These files support the index-based micro-level analysis, layer-level macro analysis, and normalized entropy analysis.

---

## 11. Reproducing permutation-aware head matching

To reproduce the central head-matching result, run:

```bash
python src/compute_head_matching_metrics.py \
  --in_csv data/attention_all_splits.csv \
  --direction head_to_dep \
  --split test \
  --model_family mono \
  --rels nsubj,obj,case,amod \
  --langs de,es,fr,gl,it,pt,ro \
  --layers 12 \
  --heads 12 \
  --arc_weight_agg max \
  --out_dir results/head_matching
```

Expected outputs:

```text
results/head_matching/head_matching_metrics_test_head_to_dep_mono.csv
results/head_matching/pairwise_head_matching_test_head_to_dep_mono.csv
results/head_matching/head_matching_permutations_test_head_to_dep_mono.csv
```

The expected values for the main manuscript table are:

| Relation | Index-based micro | Macro | Matched joint | Matched LOO |
|---|---:|---:|---:|---:|
| `nsubj` | `0.166` | `0.412` | `0.683` | `0.492` |
| `obj` | `0.136` | `0.352` | `0.730` | `0.579` |
| `case` | `0.057` | `0.158` | `0.566` | `0.323` |
| `amod` | `0.090` | `0.291` | `0.676` | `0.485` |

The `matched_joint` variant uses all analyzed relations in the functional signature and should be interpreted as an upper bound. The `matched_loo` variant excludes the target relation from the matching signature and is the main non-circular estimate reported in the manuscript.

---

## 12. Control analyses

The manuscript reports or discusses the following controls:

| Control | Current repository status |
|---|---|
| Excluding French/CamemBERT | available from the aggregated CSV and head-matching script by changing `--langs` |
| `head_to_dep` vs. `dep_to_head` | available from the aggregated CSV and scripts by changing `--direction` |
| Train/test stability | available from the aggregated CSV and scripts by changing `--split` |
| Monolingual vs. mBERT | available in `data/attention_all_splits.csv` when `model_family=mbert` is present |
| Two Galician BERT models | auxiliary CSVs/notebooks available in the repository; should be consolidated into a script |
| Portuguese base vs. Portuguese large | auxiliary CSVs/notebooks available in the repository; should be consolidated into a script |
| With/without Portuguese--Galician | available from pairwise CSVs; should be consolidated into a script |

A dedicated script such as `src/compute_control_analyses.py` is recommended for future versions of the repository, so that all controls can be reproduced without relying on notebooks.

---

## 13. Software environment

The repository contains a `requirements.txt`, but package versions were **not fixed in the current experimental log**.

The following packages are required by the pipeline:

```text
numpy
pandas
scipy
matplotlib
torch
transformers
tokenizers
sentencepiece
conllu
requests
```

Exact versions of Python, PyTorch, Transformers, Tokenizers, SciPy, NumPy, and Pandas were **not recorded in the current experimental log**.

For future archival reproducibility, the repository should include one of:

```text
requirements-lock.txt
environment.yml
pyproject.toml with pinned versions
```

---

## 14. Hardware

The extraction script detects CUDA automatically and reports whether the run is executed on GPU or CPU.

The exact hardware used for the manuscript experiments was **not recorded in the current experimental log**.

---

## 15. Known reproducibility limitations

The following limitations should remain explicit in the repository and in the manuscript:

1. UD treebanks are downloaded from branch archives unless manually pinned to a release or commit.
2. Exact UD release versions and commit hashes were not recorded.
3. Exact library versions were not recorded.
4. Tokenizer/model revision hashes from Hugging Face were not recorded.
5. Alignment and inference failure counts were not logged.
6. The Galician `dev` split uses `train_fallback_dev`.
7. French uses CamemBERT, a RoBERTa-family model, while the remaining main models are BERT-family models.
8. The analysis is observational and attention weights should not be interpreted as causal evidence without interventions such as head ablation.

---

## 16. Recommended repository organization

Recommended location for this file:

```text
docs/EXPERIMENTAL_LOG.md
```

Recommended supporting files:

```text
README.md
REPRODUCE_HEAD_MATCHING.md
scripts/reproduce_head_matching.sh
src/compute_head_matching_metrics.py
src/compute_generalization_metrics.py
src/run_ud_attention_eval.py
src/lang_resources.py
src/ud_attention_eval_core.py
```

---

## 17. Status

The current repository is sufficient to reproduce the central manuscript finding:

```text
index-based head comparison underestimates head-level syntactic generalization;
permutation-aware head matching recovers functional correspondence.
```

The remaining reproducibility improvements concern archival precision rather than the availability of the main computational result.
