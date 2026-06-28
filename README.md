# Head Matching Reveals Cross-Linguistic Generalization of Syntactic Attention in Monolingual Transformers

This repository contains the code, aggregated attention profiles, derived tables, control outputs, and reproducibility documentation for the study **“Head Matching Reveals Cross-Linguistic Generalization of Syntactic Attention in Monolingual Transformers.”**

The project investigates whether attention patterns associated with syntactic dependencies remain comparable across independently trained monolingual Transformer models. Its central methodological contribution is a **permutation-aware head-matching protocol**: instead of comparing attention heads by their absolute index across independently trained models, heads are aligned by functional signatures within each layer using the Hungarian algorithm.

## Overview

Transformer models encode non-trivial syntactic regularities in their internal representations and attention patterns. However, comparing individual attention heads across independently trained models is methodologically fragile because heads have no canonical functional ordering. Head 3 in one model is not guaranteed to perform the same function as head 3 in another model.

This repository supports an empirical evaluation of this problem across seven languages using Universal Dependencies (UD) relations and independent monolingual models. The analysis compares three levels of correspondence:

1. **Index-based micro correlation**: heads are compared by absolute layer-head position.
2. **Macro correlation**: attention is aggregated by layer.
3. **Permutation-aware matched micro correlation**: heads are aligned by functional signatures before computing cross-linguistic correspondence.

The main result is that index-based comparison underestimates head-level syntactic generalization. Once heads are matched by function, micro-level correspondence rises to the same range as, or above, layer-level consistency.

## Main Finding

The main result reproduced by this repository corresponds to the final evidence table used in the manuscript.

| Relation | Index-based micro | Macro layer | Matched micro, LOO [95% CI] | Improvement Δ [95% CI] | p | Matched LOO w/o PT–GL |
|---|---:|---:|---:|---:|---:|---:|
| `nsubj` | 0.172 | 0.412 | 0.500 [0.458, 0.554] | 0.328 [0.277, 0.377] | < .001 | 0.479 |
| `obj`   | 0.128 | 0.352 | 0.571 [0.516, 0.627] | 0.443 [0.371, 0.506] | < .001 | 0.558 |
| `case`  | 0.067 | 0.158 | 0.321 [0.258, 0.397] | 0.254 [0.198, 0.311] | < .001 | 0.293 |
| `amod`  | 0.090 | 0.291 | 0.491 [0.424, 0.557] | 0.401 [0.319, 0.478] | < .001 | 0.472 |

These values are computed on the `test` split, using monolingual models only, in the `head_to_dep` attention direction. The leave-one-out (LOO) variant avoids circularity: when evaluating one dependency relation, the head matching is estimated using only the other relations. The joint variant is available in the reproduced outputs and should be interpreted only as an upper bound.

The final manuscript table is also stored in:

```text
revision_tables/final/final_revision_evidence_table.csv
revision_tables/final/final_revision_tables.md
```

## Languages

The experiments include six Romance languages and one typologically distinct control language:

- Portuguese (`pt`)
- Galician (`gl`)
- Spanish (`es`)
- French (`fr`)
- Italian (`it`)
- Romanian (`ro`)
- German (`de`) — typological control

## Syntactic Relations

The analysis focuses on four Universal Dependencies relations:

- `nsubj` — nominal subject
- `obj` — object
- `case` — case marking
- `amod` — adjectival modifier

These relations cover argument structure, grammatical marking, and nominal modification.

## Models

The main analysis uses independent monolingual Transformer models.

| Language | Model |
|---|---|
| Portuguese | `neuralmind/bert-base-portuguese-cased` |
| Galician | `marcosgg/bert-base-gl-cased` |
| Spanish | `dccuchile/bert-base-spanish-wwm-cased` |
| French | `camembert-base` |
| Italian | `dbmdz/bert-base-italian-xxl-cased` |
| Romanian | `dumitrescustefan/bert-base-romanian-cased-v1` |
| German | `bert-base-german-cased` |

French uses CamemBERT, a RoBERTa-family model. The manuscript therefore reports a robustness check excluding French from the aggregate metrics.

The code also supports `bert-base-multilingual-cased` and `xlm-roberta-base` for supplementary comparisons, but the main reported analysis uses monolingual models only.

## Treebanks

The project uses Universal Dependencies treebanks.

| Language | Treebank |
|---|---|
| Portuguese | `UD_Portuguese-Bosque` |
| Galician | `UD_Galician-TreeGal` |
| Spanish | `UD_Spanish-AnCora` |
| French | `UD_French-GSD` |
| Italian | `UD_Italian-ISDT` |
| Romanian | `UD_Romanian-RRT` |
| German | `UD_German-GSD` |

The extraction script can download treebanks from the Universal Dependencies GitHub repositories. Exact UD release versions or commit hashes were not recorded in the current experimental log.

## Repository Structure

```text
head-matching-reveals-cross-linguistic/
├── README.md
├── LICENSE
├── requirements.txt
├── requirements-lock.txt
├── data/
│   ├── attention_all_splits.csv
│   ├── raw/
│   ├── processed/
│   └── outputs/
│       ├── gl_marcosgg.csv
│       ├── gl_fpuentes.csv
│       ├── pt_base.csv
│       └── pt_large.csv
├── docs/
│   ├── EXPERIMENTAL_LOG.md
│   ├── REPRODUCE_HEAD_MATCHING.md
│   └── REPRODUCE_CONTROLS.md
├── figures/
├── notebooks/
├── results/
├── revision_tables/
│   └── final/
│       ├── final_revision_evidence_table.csv
│       ├── final_revision_main_table.tex
│       └── final_revision_tables.md
├── scripts/
│   ├── reproduce_head_matching.sh
│   └── reproduce_controls.sh
└── src/
    ├── run_ud_attention_eval.py
    ├── ud_attention_eval_core.py
    ├── lang_resources.py
    ├── compute_generalization_metrics.py
    ├── compute_head_matching_metrics.py
    ├── bootstrap_matched_loo_inference.py
    ├── compute_control_analyses.py
    ├── negative_control_shuffle_deprel.py
    ├── negative_control_independent_head_shuffle.py
    ├── compute_ud_arc_distance_summary.py
    └── build_generalization_report.py
```

The notebooks are retained for transparency and inspection. The reproducible pipeline is implemented in the Python scripts under `src/` and the shell wrappers under `scripts/`.

## Main Scripts

### `src/run_ud_attention_eval.py`

Runs the extraction pipeline. It loads UD treebanks and Hugging Face Transformer models, aligns UD tokens with subword tokenization, extracts attention matrices, and exports aggregated attention values.

Main output fields include:

- `mean_attention`
- `std_attention`
- `n_arcs`
- `lang`
- `model_id`
- `model_family`
- `treebank`
- `split`
- `deprel`
- `layer`
- `head`
- `direction`

### `src/ud_attention_eval_core.py`

Contains the core functions for parsing CoNLL-U files, reconstructing sentence text and token spans, aligning words to subword spans using tokenizer offsets, and computing mean attention between syntactic head and dependent spans.

### `src/lang_resources.py`

Defines the language-resource catalog, including UD treebank repositories, filename prefixes, recommended Hugging Face models, supplementary models, and multilingual baselines.

### `src/compute_generalization_metrics.py`

Computes index-based generalization metrics:

- micro-level Spearman correlation across 144 layer-head positions;
- macro-level Spearman correlation across 12 layer vectors;
- normalized entropy;
- auxiliary composite score.

### `src/compute_head_matching_metrics.py`

Computes the permutation-aware head-matching analysis. It generates:

- index-based micro correlation;
- macro layer correlation;
- matched micro correlation using the joint variant;
- matched micro correlation using leave-one-out;
- pairwise language correlations;
- head permutations per language pair, layer, and relation.

This script implements the Hungarian algorithm through `scipy.optimize.linear_sum_assignment`.

### `src/bootstrap_matched_loo_inference.py`

Computes bootstrap confidence intervals and paired improvement tests for the matched LOO correlations.

### `src/compute_control_analyses.py`

Runs the main control analyses, including robustness to model family, split stability, same-language model comparison when auxiliary files are available, and related controls.

### `src/negative_control_shuffle_deprel.py`

Runs the UD-label shuffle negative control.

### `src/negative_control_independent_head_shuffle.py`

Runs the independent head-shuffle negative control.

### `src/compute_ud_arc_distance_summary.py`

Computes distance-stratified summaries for UD arcs and supports the long-distance control analyses.

### `src/build_generalization_report.py`

Builds Markdown and HTML reports from the generalization and entropy metrics.

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For stricter reproducibility, use the version-pinned environment file when available:

```bash
pip install -r requirements-lock.txt
```

Main dependencies include:

- `numpy`
- `pandas`
- `matplotlib`
- `scipy`
- `transformers`
- `torch`
- `tokenizers`
- `sentencepiece`
- `conllu`
- `tqdm`
- `jupyter`
- `notebook`

## Reproducing the Main Head-Matching Result

The repository includes the aggregated attention file:

```text
data/attention_all_splits.csv
```

This file is sufficient to reproduce the central index-based, macro-level, and head-matched metrics without re-running all Transformer models.

Run:

```bash
bash scripts/reproduce_head_matching.sh data/attention_all_splits.csv results
```

Equivalent direct command:

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
  --out_dir results
```

Expected outputs:

```text
results/head_matching_metrics_test_head_to_dep_mono.csv
results/pairwise_head_matching_test_head_to_dep_mono.csv
results/head_matching_permutations_test_head_to_dep_mono.csv
```

## Reproducing Index-Based Metrics and Entropy

Run:

```bash
python src/compute_generalization_metrics.py \
  --in_csv data/attention_all_splits.csv \
  --direction head_to_dep \
  --splits test \
  --rels nsubj,obj,case,amod \
  --model_family mono \
  --out_dir results
```

Expected outputs:

```text
results/generalization_metrics_by_deprel.csv
results/entropy_by_lang.csv
```

## Reproducing Control Analyses

Run:

```bash
bash scripts/reproduce_controls.sh
```

Expected output directory:

```text
results/controls/
```

The same-language Galician model-instance control requires the following auxiliary files:

```text
data/outputs/gl_marcosgg.csv
data/outputs/gl_fpuentes.csv
```

If the Portuguese same-language supplementary control is used, the following auxiliary files are also expected:

```text
data/outputs/pt_base.csv
data/outputs/pt_large.csv
```

## Regenerating Attention Outputs

To regenerate the aggregated attention CSV from the UD treebanks and Hugging Face models, run:

```bash
python src/run_ud_attention_eval.py \
  --langs pt,gl,es,it,fr,ro \
  --include_control \
  --control_langs de \
  --splits train,dev,test \
  --model_mode mono_vs_mbert \
  --max_sents 1000 \
  --rel_filter nsubj,obj,amod,case \
  --max_len 512 \
  --out_csv data/attention_all_splits.csv
```

For the manuscript's main monolingual analysis, downstream scripts should filter:

```text
model_family = mono
```

The extraction script computes both attention directions:

- `head_to_dep`
- `dep_to_head`

The manuscript reports the main results using `head_to_dep`.

## Experimental Design Notes

- The main analysis uses independent monolingual models.
- The maximum number of sentences per split is 1000.
- The maximum tokenized sequence length is 512 subwords.
- Galician does not have a development split in the used treebank; the pipeline marks the fallback as `train_fallback_dev`.
- French uses CamemBERT, a RoBERTa-family model, and is treated as a robustness variable.
- German is included as a typologically distinct control.
- Attention is treated as an observational signal, not as causal evidence of syntactic processing.

## Methodological Caution

This repository supports an observational interpretability study. The results provide evidence of recoverable functional correspondence among attention heads under the proposed matching protocol, but they do not prove that the matched heads are causally necessary for syntactic behavior.

For causal validation, additional interventions such as head ablation, activation patching, or causal head gating would be required.

## Known Reproducibility Caveats

The current repository substantially improves reproducibility of the main result, but some caveats remain:

- Exact UD release versions were not recorded in the current experimental log.
- Hugging Face model revisions were not recorded in the current experimental log.
- Hardware information was not recorded in the current experimental log.
- Counts of discarded alignment or inference failures were not recorded in the current experimental log.
- Alignment and inference failures are skipped by the extraction script and should ideally be logged in future runs.
- Some supplementary analyses may depend on auxiliary outputs under `data/outputs/`.

## Documentation

Additional documentation is available in:

```text
docs/EXPERIMENTAL_LOG.md
docs/REPRODUCE_HEAD_MATCHING.md
docs/REPRODUCE_CONTROLS.md
```

## Notebooks

The notebooks are retained for transparency and inspection:

- `notebooks/notebook_experimento_final_en.ipynb`
- `notebooks/notebook_respostas_revisores.ipynb`
- `notebooks/controls_supplementary.ipynb`

They should be treated as exploratory and explanatory artifacts. The official reproducibility path is through the scripts in `src/` and `scripts/`.

## Data Availability

The repository includes the aggregated attention file:

```text
data/attention_all_splits.csv
```

This file is sufficient to reproduce the central index-based, macro-level, and head-matched metrics without re-running all Transformer models.

The original UD treebanks are publicly available from the Universal Dependencies project. The extraction script can download and prepare the required treebanks automatically.

## Suggested Citation

If you use this repository, please cite the corresponding paper:

```bibtex
@article{OliveiraClaro2026HeadMatching,
  title  = {Head Matching Reveals Cross-Linguistic Generalization of Syntactic Attention in Monolingual Transformers},
  author = {Oliveira, Ricardo Gomes de and Claro, Daniela Barreiro},
  year   = {2026},
  note   = {Manuscript under review}
}
```

## License

This project is distributed under the terms of the MIT License. See the `LICENSE` file for details.

## Contact

For questions about the repository or experimental pipeline, please contact the corresponding author listed in the manuscript.
