<p align="center">
  <img src="2.png" width="600">
</p>

# Head Matching Reveals Cross-Linguistic Generalization of Syntactic Attention in Monolingual Transformers

This repository contains the code, data outputs, notebooks, and reproducibility scripts for the study **"Head Matching Reveals Cross-Linguistic Generalization of Syntactic Attention in Monolingual Transformers"**.

The project investigates whether attention patterns associated with syntactic dependencies remain comparable across independently trained monolingual Transformer models. The central methodological contribution is a **permutation-aware head-matching protocol**: instead of comparing attention heads by their absolute index across models, heads are aligned by functional signatures within each layer using the Hungarian algorithm.

## Overview

Transformer models often encode syntactic regularities in their internal representations and attention patterns. However, comparing individual attention heads across independently trained models is methodologically fragile because heads have no canonical functional ordering. Head 3 in one model is not guaranteed to perform the same function as head 3 in another model.

This repository supports an empirical evaluation of this problem across seven languages using Universal Dependencies (UD) relations and independent monolingual models. The analysis compares:

1. **Index-based micro correlation**: heads are compared by absolute layer-head position.
2. **Macro correlation**: attention is aggregated by layer.
3. **Permutation-aware matched micro correlation**: heads are aligned by functional signatures before computing cross-linguistic correspondence.

The results show that index-based comparison underestimates head-level syntactic generalization. Once heads are matched by function, micro-level correspondence rises to the same range as, or above, layer-level consistency.

## Main Finding

The main result reproduced by this repository is:

| Relation | Index-based micro | Macro layer | Matched micro, joint | Matched micro, LOO |
|---|---:|---:|---:|---:|
| `nsubj` | 0.166 | 0.412 | 0.683 | 0.492 |
| `obj`   | 0.136 | 0.352 | 0.730 | 0.579 |
| `case`  | 0.057 | 0.158 | 0.566 | 0.323 |
| `amod`  | 0.090 | 0.291 | 0.676 | 0.485 |

These values are computed on the `test` split, using monolingual models only, in the `head_to_dep` attention direction.

The leave-one-out (LOO) variant avoids circularity: when evaluating one dependency relation, the head matching is estimated using only the other relations. The joint variant uses all relations and should be interpreted as an upper bound.

## Languages

The experiments include six Romance languages and one typologically distinct control language:

- Portuguese (`pt`)
- Galician (`gl`)
- Spanish (`es`)
- Italian (`it`)
- French (`fr`)
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

The project uses independent monolingual Transformer models:

| Language | Model |
|---|---|
| Portuguese | `neuralmind/bert-base-portuguese-cased` |
| Galician | `marcosgg/bert-base-gl-cased` |
| Spanish | `dccuchile/bert-base-spanish-wwm-cased` |
| French | `camembert-base` |
| Italian | `dbmdz/bert-base-italian-xxl-cased` |
| Romanian | `dumitrescustefan/bert-base-romanian-cased-v1` |
| German | `bert-base-german-cased` |

French uses CamemBERT, a RoBERTa-family model. The paper therefore reports a robustness check excluding French from the aggregate metrics.

The code also supports `bert-base-multilingual-cased` and `xlm-roberta-base` for supplementary comparisons, but the main reported analysis uses monolingual models only.

## Treebanks

The project uses Universal Dependencies treebanks:

| Language | Treebank |
|---|---|
| Portuguese | `UD_Portuguese-Bosque` |
| Galician | `UD_Galician-TreeGal` |
| Spanish | `UD_Spanish-AnCora` |
| French | `UD_French-GSD` |
| Italian | `UD_Italian-ISDT` |
| Romanian | `UD_Romanian-RRT` |
| German | `UD_German-GSD` |

The current code downloads treebanks from the Universal Dependencies GitHub repositories. For strict archival reproducibility, the UD release version or commit hash should be recorded in an experimental log.

## Repository Structure

```text
cross-lingual-syntactic-attention/
├── README.md
├── requirements.txt
├── LICENSE
├── data/
│   ├── attention_all_splits.csv
│   ├── raw/
│   ├── processed/
│   └── outputs/
│       ├── gl_fpuentes.csv
│       ├── gl_marcosgg.csv
│       ├── pt_base.csv
│       └── pt_large.csv
├── docs/
├── figures/
├── notebooks/
│   ├── controls_supplementary.ipynb
│   ├── notebook_experimento_final_en.ipynb
│   └── notebook_respostas_revisores.ipynb
├── results/
└── src/
    ├── run_ud_attention_eval.py
    ├── ud_attention_eval_core.py
    ├── lang_resources.py
    ├── compute_generalization_metrics.py
    ├── compute_head_matching_metrics.py
    ├── build_generalization_report.py
    ├── REPRODUCE_HEAD_MATCHING.md
    └── reproduce_head_matching.sh
```

The notebooks are included for inspection and exploratory analysis. The reproducible pipeline is implemented in the Python scripts under `src/`.

## Main Scripts

### `src/run_ud_attention_eval.py`

Runs the extraction pipeline. It loads UD treebanks and Hugging Face Transformer models, aligns UD tokens with subword tokenization, extracts attention matrices, and exports aggregated attention values.

Main outputs include:

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

Contains the core functions for:

- parsing CoNLL-U files;
- reconstructing sentence text and token spans;
- aligning words to subword spans using tokenizer offsets;
- computing mean attention between syntactic head and dependent spans.

### `src/lang_resources.py`

Defines the language-resource catalog:

- UD treebank repositories;
- filename prefixes;
- recommended Hugging Face models;
- optional supplementary models;
- multilingual baselines.

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

### `src/build_generalization_report.py`

Builds Markdown and HTML reports from the generalization and entropy metrics.

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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

For stricter reproducibility, it is recommended to create a version-pinned `requirements-lock.txt` or `environment.yml`.

## Reproducing the Main Head-Matching Result

The repository includes `data/attention_all_splits.csv`, which contains the aggregated attention outputs used to reproduce the main result.

Run:

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

This generates:

```text
results/head_matching_metrics_test_head_to_dep_mono.csv
results/pairwise_head_matching_test_head_to_dep_mono.csv
results/head_matching_permutations_test_head_to_dep_mono.csv
```

The first file reproduces the central table of the paper.

You may also run:

```bash
bash scripts/reproduce_head_matching.sh
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

This generates:

```text
results/generalization_metrics_by_deprel.csv
results/entropy_by_lang.csv
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

For the paper's main monolingual analysis, downstream scripts should filter:

```text
model_family = mono
```

The extraction script computes both attention directions:

- `head_to_dep`
- `dep_to_head`

The paper reports the main results using `head_to_dep`.

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

- UD treebanks are downloaded from GitHub branches unless a release or commit hash is manually fixed.
- Library versions are not pinned in `requirements.txt`.
- Alignment and inference failures are skipped by the extraction script and should ideally be logged in a future version.
- Some supplementary controls are documented in notebooks rather than fully converted into standalone scripts.

## Notebooks

The notebooks are retained for transparency and inspection:

- `notebooks/notebook_experimento_final_en.ipynb`
- `notebooks/notebook_respostas_revisores.ipynb`
- `notebooks/controls_supplementary.ipynb`

They should be treated as exploratory and explanatory artifacts. The official reproducibility path is through the scripts in `src/`.

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
