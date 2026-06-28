#!/usr/bin/env python3
"""
Negative control for permutation-aware head matching.

This script shuffles UD relation labels within each language while preserving:
  - language
  - layer
  - head
  - model family
  - split
  - attention direction
  - the original layer-head attention matrices

The goal is to test whether matched LOO correlations remain high when the
association between attention profiles and UD relations is broken.

Input:
  data/attention_all_splits.csv

Output:
  revision_tables/negative_control_shuffle_deprel_summary.csv
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata


DEPS = ["nsubj", "obj", "case", "amod"]


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]

    if len(a) < 3:
        return float("nan")

    ra = rankdata(a, method="average")
    rb = rankdata(b, method="average")

    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")

    return float(np.corrcoef(ra, rb)[0, 1])


def weighted_mean(values: list[float], weights: list[float]) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)

    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)

    if not np.any(mask):
        return float("nan")

    return float(np.sum(v[mask] * w[mask]) / np.sum(w[mask]))


def build_tensors(df: pd.DataFrame, deps: list[str]) -> tuple[dict[str, np.ndarray], dict[tuple[str, str], int]]:
    """
    Returns:
      tensors[lang] = array [dep, layer, head]
      counts[(lang, dep)] = n_arcs
    """
    langs = sorted(df["lang"].unique())
    tensors = {}
    counts = {}

    for lang in langs:
        sub_lang = df[df["lang"] == lang]
        arr = np.full((len(deps), 12, 12), np.nan, dtype=float)

        for d_idx, dep in enumerate(deps):
            sub = sub_lang[sub_lang["deprel"] == dep]

            if sub.empty:
                continue

            # n_arcs should be constant across layer/head for a given lang/dep/split/direction
            counts[(lang, dep)] = int(sub["n_arcs"].max())

            for _, r in sub.iterrows():
                layer = int(r["layer"]) - 1
                head = int(r["head"]) - 1
                arr[d_idx, layer, head] = float(r["mean_attention"])

        tensors[lang] = arr

    return tensors, counts


def layer_permutations(TA: np.ndarray, TB: np.ndarray, keep_dims: list[int]) -> list[np.ndarray]:
    """
    Match heads in B to heads in A independently for each layer.
    TA/TB shape: [dep, layer, head]
    Returns list of 12 permutations. Each permutation maps A-head order to B-head indices.
    """
    perms = []

    for layer in range(12):
        # signatures: [head, dep_signature]
        A_sig = TA[keep_dims, layer, :].T
        B_sig = TB[keep_dims, layer, :].T

        C = np.zeros((12, 12), dtype=float)

        for i in range(12):
            for j in range(12):
                diff = A_sig[i] - B_sig[j]
                C[i, j] = float(np.sqrt(np.nansum(diff * diff)))

        row_ind, col_ind = linear_sum_assignment(C)

        # ensure order by A head index
        perm = np.empty(12, dtype=int)
        for r, c in zip(row_ind, col_ind):
            perm[r] = c

        perms.append(perm)

    return perms


def apply_permutation(B_dep: np.ndarray, perms: list[np.ndarray]) -> np.ndarray:
    """
    B_dep shape: [layer, head]
    Reorder B heads to A-head order.
    """
    out = np.full_like(B_dep, np.nan)

    for layer in range(12):
        out[layer, :] = B_dep[layer, perms[layer]]

    return out


def compute_pairwise_metrics(
    tensors: dict[str, np.ndarray],
    counts: dict[tuple[str, str], int],
    deps: list[str],
) -> pd.DataFrame:
    rows = []
    langs = sorted(tensors.keys())

    for lang_a, lang_b in combinations(langs, 2):
        TA = tensors[lang_a]
        TB = tensors[lang_b]

        for d_idx, dep in enumerate(deps):
            A = TA[d_idx]
            B = TB[d_idx]

            rho_idx = spearman_corr(A.flatten(), B.flatten())

            macro_a = np.nanmean(A, axis=1)
            macro_b = np.nanmean(B, axis=1)
            rho_macro = spearman_corr(macro_a, macro_b)

            keep = [i for i in range(len(deps)) if i != d_idx]
            perms = layer_permutations(TA, TB, keep)
            B_loo = apply_permutation(B, perms)
            rho_loo = spearman_corr(A.flatten(), B_loo.flatten())

            w = min(counts.get((lang_a, dep), 0), counts.get((lang_b, dep), 0))

            rows.append({
                "lang_a": lang_a,
                "lang_b": lang_b,
                "deprel": dep,
                "weight": w,
                "rho_micro_index": rho_idx,
                "rho_macro": rho_macro,
                "rho_micro_matched_loo": rho_loo,
            })

    return pd.DataFrame(rows)


def summarize_pairwise(pair_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for dep, sub in pair_df.groupby("deprel", sort=False):
        w = sub["weight"].to_numpy(dtype=float)

        rows.append({
            "deprel": dep,
            "rho_micro_index": weighted_mean(sub["rho_micro_index"].tolist(), w.tolist()),
            "rho_macro": weighted_mean(sub["rho_macro"].tolist(), w.tolist()),
            "rho_micro_matched_loo": weighted_mean(sub["rho_micro_matched_loo"].tolist(), w.tolist()),
        })

    return pd.DataFrame(rows)


def shuffle_relation_labels(
    tensors: dict[str, np.ndarray],
    counts: dict[tuple[str, str], int],
    deps: list[str],
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], dict[tuple[str, str], int]]:
    """
    Independently permute dependency-label assignments within each language.
    This breaks cross-linguistic relation identity while preserving each language's
    layer-head profiles.
    """
    shuffled_tensors = {}
    shuffled_counts = {}

    for lang, T in tensors.items():
        perm = rng.permutation(len(deps))

        # New label dep_i receives old matrix dep_perm[i]
        shuffled_tensors[lang] = T[perm, :, :].copy()

        for new_i, old_i in enumerate(perm):
            new_dep = deps[new_i]
            old_dep = deps[old_i]
            shuffled_counts[(lang, new_dep)] = counts.get((lang, old_dep), 0)

    return shuffled_tensors, shuffled_counts


def percentile_ci(x: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/attention_all_splits.csv")
    ap.add_argument("--output", default="revision_tables/negative_control_shuffle_deprel_summary.csv")
    ap.add_argument("--split", default="test")
    ap.add_argument("--direction", default="head_to_dep")
    ap.add_argument("--model_family", default="mono")
    ap.add_argument("--n_iter", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260627)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(args.input)

    df = df[
        (df["split"] == args.split)
        & (df["direction"] == args.direction)
        & (df["model_family"] == args.model_family)
        & (df["deprel"].isin(DEPS))
    ].copy()

    print("Filtered shape:", df.shape)
    print("Languages:", sorted(df["lang"].unique()))
    print("Relations:", sorted(df["deprel"].unique()))

    tensors, counts = build_tensors(df, DEPS)

    real_pairwise = compute_pairwise_metrics(tensors, counts, DEPS)
    real_summary = summarize_pairwise(real_pairwise)

    null_rows = []

    for i in range(args.n_iter):
        shuf_tensors, shuf_counts = shuffle_relation_labels(tensors, counts, DEPS, rng)
        shuf_pairwise = compute_pairwise_metrics(shuf_tensors, shuf_counts, DEPS)
        shuf_summary = summarize_pairwise(shuf_pairwise)

        shuf_summary["iteration"] = i
        null_rows.append(shuf_summary)

    null_df = pd.concat(null_rows, ignore_index=True)

    rows = []

    for _, r in real_summary.iterrows():
        dep = r["deprel"]
        null_sub = null_df[null_df["deprel"] == dep]

        null_vals = null_sub["rho_micro_matched_loo"].to_numpy(dtype=float)
        null_lo, null_hi = percentile_ci(null_vals)

        real_val = float(r["rho_micro_matched_loo"])

        p_null_ge_real = float((1 + np.sum(null_vals >= real_val)) / (len(null_vals) + 1))

        rows.append({
            "deprel": dep,
            "real_rho_micro_index": float(r["rho_micro_index"]),
            "real_rho_macro": float(r["rho_macro"]),
            "real_rho_micro_matched_loo": real_val,
            "null_mean_rho_micro_matched_loo": float(np.mean(null_vals)),
            "null_ci_low": null_lo,
            "null_ci_high": null_hi,
            "real_minus_null_mean": real_val - float(np.mean(null_vals)),
            "p_null_ge_real": p_null_ge_real,
            "n_null_iterations": len(null_vals),
        })

    out = pd.DataFrame(rows)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    real_pairwise.to_csv(Path(args.output).with_name("negative_control_real_pairwise_recomputed.csv"), index=False)
    null_df.to_csv(Path(args.output).with_name("negative_control_shuffle_deprel_null_iterations.csv"), index=False)

    print("[OK] wrote", args.output)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
