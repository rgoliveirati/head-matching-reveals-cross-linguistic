#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_head_matching_metrics.py
================================

Permutation-aware head matching for cross-linguistic syntactic-attention profiles.

This script reproduces the central comparison used in the revised manuscript:
absolute-index micro correlation vs. layer-level macro correlation vs. head-matched
micro correlation, including the joint and leave-one-out (LOO) variants.

Input
-----
A CSV produced by run_ud_attention_eval.py, with at least the following columns:
    lang, deprel, layer, head, direction, mean_attention, n_arcs
Optional but supported:
    model_family, split, model_id, treebank

Outputs
-------
1. head_matching_metrics_<split>.csv
   One row per dependency relation, with weighted aggregate micro correlations:
       rho_micro_index
       rho_macro_mean
       rho_macro_weighted
       rho_micro_matched_joint
       rho_micro_matched_loo

2. pairwise_head_matching_<split>.csv
   One row per language pair and dependency relation, with pairwise correlations.

3. head_matching_permutations_<split>.csv
   The head assignments selected by the Hungarian algorithm.

Method
------
For each language and dependency relation, the script builds a 12 x 12 attention
matrix M[layer, head]. For each pair of languages, heads are matched independently
within each layer. The cost of assigning head i in language A to head j in language B
is the Euclidean distance between their functional signatures across dependency
relations.

Two matched variants are reported:
    matched_joint: uses all relations to estimate the matching;
    matched_loo:   when evaluating relation d, estimates the matching using all
                   other relations, avoiding circularity.

The micro aggregate uses the same weighted scheme described in the article:
for each language pair, the weight is min(n_arcs(lang_a, dep), n_arcs(lang_b, dep)).
By default, n_arcs is read with --arc_weight_agg max because run_ud_attention_eval.py
stores the same arc count once per layer-head cell. Use --arc_weight_agg sum to
match older scripts that summed repeated cells.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

REQUIRED_COLUMNS = {"lang", "deprel", "layer", "head", "direction", "mean_attention", "n_arcs"}


@dataclass(frozen=True)
class Config:
    layers: int
    heads: int
    deps: List[str]
    langs: List[str]
    arc_weight_agg: str


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation implemented through average ranks and Pearson."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size != b.size or a.size == 0:
        return float("nan")
    ra = pd.Series(a).rank(method="average").to_numpy(dtype=float)
    rb = pd.Series(b).rank(method="average").to_numpy(dtype=float)
    if np.allclose(ra, ra[0]) or np.allclose(rb, rb[0]):
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def normalize_csv_list(value: str, available: Sequence[str]) -> List[str]:
    """Resolve comma-separated CLI values; '(auto)' means use available values."""
    value = (value or "").strip()
    if value in {"", "(auto)", "auto", "(todos)", "todos"}:
        return list(available)
    requested = [x.strip() for x in value.split(",") if x.strip()]
    missing = sorted(set(requested) - set(available))
    if missing:
        raise SystemExit(f"Values not found in input CSV: {missing}. Available: {list(available)}")
    return requested


def load_and_filter(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.in_csv)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise SystemExit(f"Invalid CSV. Missing columns: {sorted(missing)}")

    df = df.copy()
    df["lang"] = df["lang"].astype(str)
    df["deprel"] = df["deprel"].astype(str)
    df["direction"] = df["direction"].astype(str)
    df["layer"] = pd.to_numeric(df["layer"], errors="coerce").astype("Int64")
    df["head"] = pd.to_numeric(df["head"], errors="coerce").astype("Int64")
    df["mean_attention"] = pd.to_numeric(df["mean_attention"], errors="coerce")
    df["n_arcs"] = pd.to_numeric(df["n_arcs"], errors="coerce")
    df = df.dropna(subset=["layer", "head", "mean_attention", "n_arcs"])
    df["layer"] = df["layer"].astype(int)
    df["head"] = df["head"].astype(int)

    df = df[df["direction"] == args.direction].copy()

    if args.split not in {"", "(todos)", "todos", "all", "(all)"}:
        if "split" not in df.columns:
            raise SystemExit("--split was provided, but input CSV has no 'split' column.")
        wanted = [x.strip() for x in args.split.split(",") if x.strip()]
        df = df[df["split"].astype(str).isin(wanted)].copy()

    if args.model_family not in {"", "(todos)", "todos", "all", "(all)"}:
        if "model_family" not in df.columns:
            raise SystemExit("--model_family was provided, but input CSV has no 'model_family' column.")
        wanted = [x.strip() for x in args.model_family.split(",") if x.strip()]
        df = df[df["model_family"].astype(str).isin(wanted)].copy()

    if df.empty:
        raise SystemExit("DataFrame is empty after filters.")

    # Collapse duplicate cells, if any, after filtering. This protects against accidental
    # duplicate rows and against CSVs concatenated from repeated runs.
    group_cols = ["lang", "deprel", "layer", "head", "direction"]
    keep_cols = group_cols + ["mean_attention", "n_arcs"]
    df = df[keep_cols].groupby(group_cols, as_index=False).agg(
        mean_attention=("mean_attention", "mean"),
        n_arcs=("n_arcs", "max"),
    )
    return df


def make_config(df: pd.DataFrame, args: argparse.Namespace) -> Config:
    available_deps = sorted(df["deprel"].unique().tolist())
    available_langs = sorted(df["lang"].unique().tolist())
    deps = normalize_csv_list(args.rels, available_deps)
    langs = normalize_csv_list(args.langs, available_langs)
    df_deps = set(df["deprel"].unique())
    df_langs = set(df["lang"].unique())
    if len(deps) < 2:
        print("[WARN] LOO matching needs at least two relations; matched_loo will be NaN.")
    if len(langs) < 2:
        raise SystemExit("At least two languages are required.")
    L = int(args.layers or df["layer"].max())
    H = int(args.heads or df["head"].max())
    return Config(layers=L, heads=H, deps=deps, langs=langs, arc_weight_agg=args.arc_weight_agg)


def matrix_for(df: pd.DataFrame, cfg: Config, lang: str, dep: str) -> np.ndarray:
    M = np.zeros((cfg.layers, cfg.heads), dtype=float)
    sub = df[(df["lang"] == lang) & (df["deprel"] == dep)]
    for _, r in sub.iterrows():
        l = int(r["layer"]) - 1
        h = int(r["head"]) - 1
        if 0 <= l < cfg.layers and 0 <= h < cfg.heads:
            M[l, h] = float(r["mean_attention"])
    return M


def layer_vector_for(df: pd.DataFrame, cfg: Config, lang: str, dep: str) -> np.ndarray:
    M = matrix_for(df, cfg, lang, dep)
    return M.mean(axis=1)


def signature_tensor(df: pd.DataFrame, cfg: Config, lang: str) -> np.ndarray:
    """Return tensor [layers, heads, deps] for one language."""
    T = np.zeros((cfg.layers, cfg.heads, len(cfg.deps)), dtype=float)
    for i, dep in enumerate(cfg.deps):
        T[:, :, i] = matrix_for(df, cfg, lang, dep)
    return T


def arc_count(df: pd.DataFrame, cfg: Config, lang: str, dep: str) -> float:
    sub = df[(df["lang"] == lang) & (df["deprel"] == dep)]
    if sub.empty:
        return 0.0
    vals = sub["n_arcs"].to_numpy(dtype=float)
    if cfg.arc_weight_agg == "sum":
        return float(np.nansum(vals))
    if cfg.arc_weight_agg == "mean":
        return float(np.nanmean(vals))
    if cfg.arc_weight_agg == "median":
        return float(np.nanmedian(vals))
    # default: max, because n_arcs is repeated by layer-head cell in the pipeline output.
    return float(np.nanmax(vals))


def layer_permutations(TA: np.ndarray, TB: np.ndarray, keep_dims: Sequence[int]) -> List[np.ndarray]:
    """For each layer, match heads in B to heads in A using Euclidean signature distance.

    Returns a list of arrays. perms[l][i] is the head index in B assigned to head i in A.
    """
    if len(keep_dims) == 0:
        raise ValueError("keep_dims cannot be empty")
    perms: List[np.ndarray] = []
    for layer in range(TA.shape[0]):
        A = TA[layer][:, keep_dims]
        B = TB[layer][:, keep_dims]
        C = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
        row_ind, col_ind = linear_sum_assignment(C)
        # linear_sum_assignment returns sorted row indices for square matrices, but make explicit.
        perm = np.empty(TA.shape[1], dtype=int)
        perm[row_ind] = col_ind
        perms.append(perm)
    return perms


def matched_matrix(TB: np.ndarray, dep_index: int, perms: Sequence[np.ndarray]) -> np.ndarray:
    M = np.zeros((TB.shape[0], TB.shape[1]), dtype=float)
    for layer, perm in enumerate(perms):
        M[layer, :] = TB[layer, perm, dep_index]
    return M


def pair_correlations(df: pd.DataFrame, cfg: Config, lang_a: str, lang_b: str) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    TA = signature_tensor(df, cfg, lang_a)
    TB = signature_tensor(df, cfg, lang_b)

    joint_perms: Optional[List[np.ndarray]] = None
    if len(cfg.deps) >= 1:
        joint_perms = layer_permutations(TA, TB, list(range(len(cfg.deps))))

    rows: List[Dict[str, object]] = []
    perm_rows: List[Dict[str, object]] = []

    # Store joint permutations once per pair.
    if joint_perms is not None:
        for layer, perm in enumerate(joint_perms, start=1):
            for head_a, head_b0 in enumerate(perm, start=1):
                perm_rows.append({
                    "lang_a": lang_a,
                    "lang_b": lang_b,
                    "mode": "matched_joint",
                    "deprel_eval": "(all)",
                    "layer": layer,
                    "head_a": head_a,
                    "head_b": int(head_b0) + 1,
                })

    for dep_index, dep in enumerate(cfg.deps):
        A = TA[:, :, dep_index]
        B_index = TB[:, :, dep_index]
        rho_index = spearman_corr(A.flatten(), B_index.flatten())

        B_joint = matched_matrix(TB, dep_index, joint_perms) if joint_perms is not None else np.full_like(A, np.nan)
        rho_joint = spearman_corr(A.flatten(), B_joint.flatten())

        if len(cfg.deps) > 1:
            keep = [i for i in range(len(cfg.deps)) if i != dep_index]
            loo_perms = layer_permutations(TA, TB, keep)
            B_loo = matched_matrix(TB, dep_index, loo_perms)
            rho_loo = spearman_corr(A.flatten(), B_loo.flatten())
            for layer, perm in enumerate(loo_perms, start=1):
                for head_a, head_b0 in enumerate(perm, start=1):
                    perm_rows.append({
                        "lang_a": lang_a,
                        "lang_b": lang_b,
                        "mode": "matched_loo",
                        "deprel_eval": dep,
                        "layer": layer,
                        "head_a": head_a,
                        "head_b": int(head_b0) + 1,
                    })
        else:
            rho_loo = float("nan")

        macro_a = layer_vector_for(df, cfg, lang_a, dep)
        macro_b = layer_vector_for(df, cfg, lang_b, dep)
        rho_macro = spearman_corr(macro_a, macro_b)
        weight = min(arc_count(df, cfg, lang_a, dep), arc_count(df, cfg, lang_b, dep))

        rows.append({
            "lang_a": lang_a,
            "lang_b": lang_b,
            "deprel": dep,
            "weight_min_n_arcs": weight,
            "rho_micro_index": rho_index,
            "rho_macro": rho_macro,
            "rho_micro_matched_joint": rho_joint,
            "rho_micro_matched_loo": rho_loo,
        })
    return rows, perm_rows


def weighted_mean(values: Iterable[float], weights: Iterable[float]) -> float:
    vals = np.asarray(list(values), dtype=float)
    w = np.asarray(list(weights), dtype=float)
    ok = ~np.isnan(vals) & ~np.isnan(w) & (w > 0)
    if not ok.any():
        return float("nan")
    return float(np.sum(vals[ok] * w[ok]) / np.sum(w[ok]))


def summarize_pairs(pair_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for dep in cfg.deps:
        sub = pair_df[pair_df["deprel"] == dep].copy()
        if sub.empty:
            continue
        weights = sub["weight_min_n_arcs"].to_numpy(dtype=float)
        rows.append({
            "deprel": dep,
            "n_langs": len(cfg.langs),
            "n_pairs": len(sub),
            "rho_micro_index": weighted_mean(sub["rho_micro_index"], weights),
            "rho_macro_mean": float(np.nanmean(sub["rho_macro"].to_numpy(dtype=float))),
            "rho_macro_weighted": weighted_mean(sub["rho_macro"], weights),
            "rho_micro_matched_joint": weighted_mean(sub["rho_micro_matched_joint"], weights),
            "rho_micro_matched_loo": weighted_mean(sub["rho_micro_matched_loo"], weights),
            "arc_weight_agg": cfg.arc_weight_agg,
        })
    return pd.DataFrame(rows)


def safe_tag(text: str) -> str:
    text = str(text or "all").replace(",", "-").replace("/", "-").replace(" ", "")
    if text in {"", "(todos)", "todos", "all", "(all)"}:
        return "all"
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute permutation-aware head-matching metrics.")
    parser.add_argument("--in_csv", required=True, help="CSV produced by run_ud_attention_eval.py")
    parser.add_argument("--direction", default="head_to_dep", help="Attention direction to analyze")
    parser.add_argument("--split", default="test", help="Split(s) to analyze, comma-separated; use all for no filtering")
    parser.add_argument("--model_family", default="mono", help="Model family filter; use all for no filtering")
    parser.add_argument("--rels", default="nsubj,obj,case,amod", help="Relations, comma-separated; use auto for all")
    parser.add_argument("--langs", default="auto", help="Languages, comma-separated; use auto for all in CSV")
    parser.add_argument("--layers", type=int, default=0, help="Number of layers; 0 = infer from CSV")
    parser.add_argument("--heads", type=int, default=0, help="Number of heads; 0 = infer from CSV")
    parser.add_argument("--arc_weight_agg", choices=["max", "sum", "mean", "median"], default="max",
                        help="How to reduce repeated n_arcs cells into one language/relation weight")
    parser.add_argument("--out_dir", default="results", help="Output directory")
    parser.add_argument("--out_csv", default="", help="Optional explicit summary CSV path")
    parser.add_argument("--out_pairs", default="", help="Optional explicit pairwise CSV path")
    parser.add_argument("--out_permutations", default="", help="Optional explicit permutations CSV path")
    parser.add_argument("--round", type=int, default=6, help="Decimal places for CSV outputs")
    args = parser.parse_args()

    df = load_and_filter(args)
    cfg = make_config(df, args)

    # Keep only selected deps/langs.
    df = df[df["deprel"].isin(cfg.deps) & df["lang"].isin(cfg.langs)].copy()
    if df.empty:
        raise SystemExit("DataFrame is empty after relation/language selection.")

    pair_rows: List[Dict[str, object]] = []
    perm_rows: List[Dict[str, object]] = []
    for lang_a, lang_b in combinations(cfg.langs, 2):
        rows, perms = pair_correlations(df, cfg, lang_a, lang_b)
        pair_rows.extend(rows)
        perm_rows.extend(perms)

    pair_df = pd.DataFrame(pair_rows)
    summary_df = summarize_pairs(pair_df, cfg)
    perm_df = pd.DataFrame(perm_rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{safe_tag(args.split)}_{safe_tag(args.direction)}_{safe_tag(args.model_family)}"

    out_summary = Path(args.out_csv) if args.out_csv else out_dir / f"head_matching_metrics_{tag}.csv"
    out_pairs = Path(args.out_pairs) if args.out_pairs else out_dir / f"pairwise_head_matching_{tag}.csv"
    out_perms = Path(args.out_permutations) if args.out_permutations else out_dir / f"head_matching_permutations_{tag}.csv"

    # Round only numeric columns for readability; preserve full structure.
    for frame in (summary_df, pair_df):
        num_cols = frame.select_dtypes(include=[np.number]).columns
        frame[num_cols] = frame[num_cols].round(args.round)

    summary_df.to_csv(out_summary, index=False, encoding="utf-8")
    pair_df.to_csv(out_pairs, index=False, encoding="utf-8")
    perm_df.to_csv(out_perms, index=False, encoding="utf-8")

    print("[OK] summary:", out_summary.resolve())
    print("[OK] pairwise:", out_pairs.resolve())
    print("[OK] permutations:", out_perms.resolve())
    print("\nSummary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
