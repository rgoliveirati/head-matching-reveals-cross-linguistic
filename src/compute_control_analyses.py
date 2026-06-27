#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_control_analyses.py
===========================

Reproducible control analyses for the cross-linguistic syntactic-attention study.

This script complements compute_head_matching_metrics.py. It is designed to
reproduce the controls discussed in the manuscript:

1. French/CamemBERT exclusion;
2. train/test stability;
3. attention-direction robustness;
4. monolingual vs. mBERT comparison;
5. aggregation with and without the Portuguese--Galician pair;
6. Romance--Romance vs. German--Romance pair comparison;
7. optional two-Galician-model control using auxiliary CSVs.

Input
-----
A CSV produced by run_ud_attention_eval.py with at least:
    lang, deprel, layer, head, direction, mean_attention, n_arcs
Optional but expected for the main controls:
    model_family, split, model_id, treebank

Outputs
-------
The script writes CSV files into --out_dir:
    without_french_control.csv
    train_test_stability.csv
    direction_control.csv
    mono_vs_mbert_control.csv
    without_pt_gl_pair.csv
    romance_vs_german_control.csv
    galician_two_models_control.csv         (if auxiliary files are provided)
    controls_manifest.csv

Method
------
All controls use the same weighted aggregation as the main head-matching analysis.
Pair weights are min(n_arcs(lang_a, dep), n_arcs(lang_b, dep)), with n_arcs reduced
by --arc_weight_agg (default: max, because n_arcs is repeated per layer-head cell in
the aggregated attention CSV).

The script reports both index-based and permutation-aware matched correlations.
Matched LOO is the preferred non-circular estimate.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

REQUIRED_COLUMNS = {"lang", "deprel", "layer", "head", "direction", "mean_attention", "n_arcs"}
METRIC_COLUMNS = [
    "rho_micro_index",
    "rho_macro_weighted",
    "rho_micro_matched_joint",
    "rho_micro_matched_loo",
]
ROMANCE = {"pt", "gl", "es", "fr", "it", "ro"}
GERMAN = "de"


def parse_csv_list(value: str, default: Sequence[str]) -> List[str]:
    value = (value or "").strip()
    if value in {"", "auto", "(auto)", "all", "(all)", "todos", "(todos)"}:
        return list(default)
    return [x.strip() for x in value.split(",") if x.strip()]


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size != b.size or a.size == 0:
        return float("nan")
    ra = pd.Series(a).rank(method="average").to_numpy(dtype=float)
    rb = pd.Series(b).rank(method="average").to_numpy(dtype=float)
    if np.allclose(ra, ra[0]) or np.allclose(rb, rb[0]):
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def weighted_mean(values: Iterable[float], weights: Iterable[float]) -> float:
    vals = np.asarray(list(values), dtype=float)
    w = np.asarray(list(weights), dtype=float)
    ok = ~np.isnan(vals) & ~np.isnan(w) & (w > 0)
    if not ok.any():
        return float("nan")
    return float(np.sum(vals[ok] * w[ok]) / np.sum(w[ok]))


def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise SystemExit(f"Invalid CSV {path}. Missing columns: {sorted(missing)}")
    df = df.copy()
    for c in ["lang", "deprel", "direction"]:
        df[c] = df[c].astype(str)
    for c in ["layer", "head"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in ["mean_attention", "n_arcs"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["layer", "head", "mean_attention", "n_arcs"])
    df["layer"] = df["layer"].astype(int)
    df["head"] = df["head"].astype(int)
    return df


def filter_df(
    df: pd.DataFrame,
    *,
    direction: Optional[str] = None,
    split: Optional[str] = None,
    model_family: Optional[str] = None,
    rels: Optional[Sequence[str]] = None,
    langs: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    out = df.copy()
    if direction not in {None, "", "all", "(all)", "todos", "(todos)"}:
        out = out[out["direction"].astype(str) == str(direction)].copy()
    if split not in {None, "", "all", "(all)", "todos", "(todos)"}:
        if "split" not in out.columns:
            raise SystemExit("A split filter was requested, but the input CSV has no 'split' column.")
        wanted = parse_csv_list(str(split), [])
        out = out[out["split"].astype(str).isin(wanted)].copy()
    if model_family not in {None, "", "all", "(all)", "todos", "(todos)"}:
        if "model_family" not in out.columns:
            raise SystemExit("A model_family filter was requested, but the input CSV has no 'model_family' column.")
        wanted = parse_csv_list(str(model_family), [])
        out = out[out["model_family"].astype(str).isin(wanted)].copy()
    if rels:
        out = out[out["deprel"].isin(list(rels))].copy()
    if langs:
        out = out[out["lang"].isin(list(langs))].copy()

    # Collapse duplicate cells after filtering.
    group_cols = ["lang", "deprel", "layer", "head", "direction"]
    out = out[group_cols + ["mean_attention", "n_arcs"]].groupby(group_cols, as_index=False).agg(
        mean_attention=("mean_attention", "mean"),
        n_arcs=("n_arcs", "max"),
    )
    return out


def infer_layers_heads(df: pd.DataFrame, layers: int, heads: int) -> Tuple[int, int]:
    L = int(layers or df["layer"].max())
    H = int(heads or df["head"].max())
    return L, H


def matrix_for(df: pd.DataFrame, lang: str, dep: str, L: int, H: int) -> np.ndarray:
    M = np.zeros((L, H), dtype=float)
    sub = df[(df["lang"] == lang) & (df["deprel"] == dep)]
    for _, r in sub.iterrows():
        l = int(r["layer"]) - 1
        h = int(r["head"]) - 1
        if 0 <= l < L and 0 <= h < H:
            M[l, h] = float(r["mean_attention"])
    return M


def tensor_for(df: pd.DataFrame, lang: str, deps: Sequence[str], L: int, H: int) -> np.ndarray:
    T = np.zeros((L, H, len(deps)), dtype=float)
    for i, dep in enumerate(deps):
        T[:, :, i] = matrix_for(df, lang, dep, L, H)
    return T


def layer_permutations(TA: np.ndarray, TB: np.ndarray, keep_dims: Sequence[int]) -> List[np.ndarray]:
    if len(keep_dims) == 0:
        raise ValueError("keep_dims cannot be empty")
    perms: List[np.ndarray] = []
    for layer in range(TA.shape[0]):
        A = TA[layer][:, keep_dims]
        B = TB[layer][:, keep_dims]
        C = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
        row_ind, col_ind = linear_sum_assignment(C)
        perm = np.empty(TA.shape[1], dtype=int)
        perm[row_ind] = col_ind
        perms.append(perm)
    return perms


def apply_permutation(TB: np.ndarray, dep_index: int, perms: Sequence[np.ndarray]) -> np.ndarray:
    M = np.zeros((TB.shape[0], TB.shape[1]), dtype=float)
    for layer, perm in enumerate(perms):
        M[layer, :] = TB[layer, perm, dep_index]
    return M


def arc_count(df: pd.DataFrame, lang: str, dep: str, agg: str = "max") -> float:
    sub = df[(df["lang"] == lang) & (df["deprel"] == dep)]
    if sub.empty:
        return 0.0
    vals = sub["n_arcs"].to_numpy(dtype=float)
    if agg == "sum":
        return float(np.nansum(vals))
    if agg == "mean":
        return float(np.nanmean(vals))
    if agg == "median":
        return float(np.nanmedian(vals))
    return float(np.nanmax(vals))


def pairwise_metrics(df: pd.DataFrame, deps: Sequence[str], langs: Sequence[str], L: int, H: int, arc_weight_agg: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for lang_a, lang_b in combinations(langs, 2):
        TA = tensor_for(df, lang_a, deps, L, H)
        TB = tensor_for(df, lang_b, deps, L, H)
        joint_perms = layer_permutations(TA, TB, list(range(len(deps)))) if deps else []
        for dep_index, dep in enumerate(deps):
            A = TA[:, :, dep_index]
            B = TB[:, :, dep_index]
            rho_index = spearman_corr(A.flatten(), B.flatten())
            macro_a = A.mean(axis=1)
            macro_b = B.mean(axis=1)
            rho_macro = spearman_corr(macro_a, macro_b)
            B_joint = apply_permutation(TB, dep_index, joint_perms)
            rho_joint = spearman_corr(A.flatten(), B_joint.flatten())
            if len(deps) > 1:
                keep = [i for i in range(len(deps)) if i != dep_index]
                loo_perms = layer_permutations(TA, TB, keep)
                B_loo = apply_permutation(TB, dep_index, loo_perms)
                rho_loo = spearman_corr(A.flatten(), B_loo.flatten())
            else:
                rho_loo = float("nan")
            rows.append({
                "lang_a": lang_a,
                "lang_b": lang_b,
                "deprel": dep,
                "weight_min_n_arcs": min(arc_count(df, lang_a, dep, arc_weight_agg), arc_count(df, lang_b, dep, arc_weight_agg)),
                "rho_micro_index": rho_index,
                "rho_macro": rho_macro,
                "rho_micro_matched_joint": rho_joint,
                "rho_micro_matched_loo": rho_loo,
            })
    return pd.DataFrame(rows)


def summarize_pairwise(pair_df: pd.DataFrame, deps: Sequence[str], n_langs: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for dep in deps:
        sub = pair_df[pair_df["deprel"] == dep]
        if sub.empty:
            continue
        w = sub["weight_min_n_arcs"].to_numpy(dtype=float)
        rows.append({
            "deprel": dep,
            "n_langs": n_langs,
            "n_pairs": len(sub),
            "rho_micro_index": weighted_mean(sub["rho_micro_index"], w),
            "rho_macro_mean": float(np.nanmean(sub["rho_macro"].to_numpy(dtype=float))),
            "rho_macro_weighted": weighted_mean(sub["rho_macro"], w),
            "rho_micro_matched_joint": weighted_mean(sub["rho_micro_matched_joint"], w),
            "rho_micro_matched_loo": weighted_mean(sub["rho_micro_matched_loo"], w),
        })
    return pd.DataFrame(rows)


def compute_summary(
    raw_df: pd.DataFrame,
    *,
    direction: str,
    split: str,
    model_family: str,
    rels: Sequence[str],
    langs: Sequence[str],
    layers: int,
    heads: int,
    arc_weight_agg: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = filter_df(raw_df, direction=direction, split=split, model_family=model_family, rels=rels, langs=langs)
    if df.empty or len(df["lang"].unique()) < 2:
        return pd.DataFrame(), pd.DataFrame()
    L, H = infer_layers_heads(df, layers, heads)
    pair_df = pairwise_metrics(df, rels, langs, L, H, arc_weight_agg)
    summary_df = summarize_pairwise(pair_df, rels, n_langs=len(langs))
    return summary_df, pair_df


def comparison_table(a: pd.DataFrame, b: pd.DataFrame, label_a: str, label_b: str, metrics: Sequence[str] = METRIC_COLUMNS) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if a.empty or b.empty:
        return pd.DataFrame(columns=["deprel", "metric", label_a, label_b, "delta", "abs_delta"])
    merged = a.merge(b, on="deprel", suffixes=(f"_{label_a}", f"_{label_b}"))
    for _, r in merged.iterrows():
        for m in metrics:
            ca = f"{m}_{label_a}"
            cb = f"{m}_{label_b}"
            if ca in merged.columns and cb in merged.columns:
                va = float(r[ca])
                vb = float(r[cb])
                rows.append({"deprel": r["deprel"], "metric": m, label_a: va, label_b: vb, "delta": vb - va, "abs_delta": abs(vb - va)})
    return pd.DataFrame(rows)


def aggregate_pair_subset(pair_df: pd.DataFrame, deps: Sequence[str], label: str, predicate) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for dep in deps:
        sub = pair_df[pair_df["deprel"] == dep].copy()
        if sub.empty:
            continue
        mask = sub.apply(lambda r: bool(predicate(str(r["lang_a"]), str(r["lang_b"]))), axis=1)
        sub = sub[mask]
        if sub.empty:
            continue
        w = sub["weight_min_n_arcs"].to_numpy(dtype=float)
        for m in ["rho_micro_index", "rho_macro", "rho_micro_matched_joint", "rho_micro_matched_loo"]:
            rows.append({
                "group": label,
                "deprel": dep,
                "metric": m,
                "n_pairs": len(sub),
                "mean": float(np.nanmean(sub[m].to_numpy(dtype=float))),
                "weighted_mean": weighted_mean(sub[m], w),
            })
    return pd.DataFrame(rows)


def make_same_language_control(path_a: str, path_b: str, label_a: str, label_b: str, direction: str, split: str, rels: Sequence[str], layers: int, heads: int, arc_weight_agg: str) -> pd.DataFrame:
    dfa = load_csv(path_a)
    dfb = load_csv(path_b)
    dfa = filter_df(dfa, direction=direction, split=split, model_family=None, rels=rels, langs=None)
    dfb = filter_df(dfb, direction=direction, split=split, model_family=None, rels=rels, langs=None)
    dfa["lang"] = label_a
    dfb["lang"] = label_b
    df = pd.concat([dfa, dfb], ignore_index=True)
    L, H = infer_layers_heads(df, layers, heads)
    pair_df = pairwise_metrics(df, rels, [label_a, label_b], L, H, arc_weight_agg)
    out = pair_df.copy()
    out.insert(0, "control", f"{label_a}_vs_{label_b}")
    return out


def round_numeric(df: pd.DataFrame, ndigits: int) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].round(ndigits)
    return out


def save_csv(df: pd.DataFrame, path: Path, ndigits: int, manifest: List[Dict[str, object]], description: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    round_numeric(df, ndigits).to_csv(path, index=False, encoding="utf-8")
    manifest.append({
        "file": str(path),
        "rows": int(len(df)),
        "status": "ok" if len(df) else "empty",
        "description": description,
    })


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute reproducible control analyses for the syntactic-attention study.")
    ap.add_argument("--attention_csv", required=True, help="Main aggregated attention CSV.")
    ap.add_argument("--out_dir", default="results/controls", help="Output directory.")
    ap.add_argument("--rels", default="nsubj,obj,case,amod")
    ap.add_argument("--langs", default="de,es,fr,gl,it,pt,ro")
    ap.add_argument("--direction", default="head_to_dep")
    ap.add_argument("--split", default="test")
    ap.add_argument("--model_family", default="mono")
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--arc_weight_agg", choices=["max", "sum", "mean", "median"], default="max")
    ap.add_argument("--gl_model_a_csv", default="", help="Optional CSV for first Galician model.")
    ap.add_argument("--gl_model_b_csv", default="", help="Optional CSV for second Galician model.")
    ap.add_argument("--gl_model_a_label", default="gl_marcosgg")
    ap.add_argument("--gl_model_b_label", default="gl_fpuentes")
    ap.add_argument("--round", type=int, default=6)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rels = parse_csv_list(args.rels, [])
    langs = parse_csv_list(args.langs, [])
    raw = load_csv(args.attention_csv)
    manifest: List[Dict[str, object]] = []

    # Baseline summary/pairwise for main setup.
    main_summary, main_pairs = compute_summary(
        raw,
        direction=args.direction,
        split=args.split,
        model_family=args.model_family,
        rels=rels,
        langs=langs,
        layers=args.layers,
        heads=args.heads,
        arc_weight_agg=args.arc_weight_agg,
    )
    save_csv(main_summary, out_dir / "main_head_matching_summary.csv", args.round, manifest, "Main mono/test/head_to_dep summary used as baseline for controls.")
    save_csv(main_pairs, out_dir / "main_pairwise_head_matching.csv", args.round, manifest, "Main pairwise correlations used for pair-level controls.")

    # 1. Excluding French/CamemBERT.
    no_fr_langs = [l for l in langs if l != "fr"]
    no_fr_summary, _ = compute_summary(
        raw, direction=args.direction, split=args.split, model_family=args.model_family,
        rels=rels, langs=no_fr_langs, layers=args.layers, heads=args.heads, arc_weight_agg=args.arc_weight_agg,
    )
    without_fr = comparison_table(main_summary, no_fr_summary, "all_7", "without_fr")
    save_csv(without_fr, out_dir / "without_french_control.csv", args.round, manifest, "All languages vs. French/CamemBERT excluded.")

    # 2. Train/test stability.
    train_summary, _ = compute_summary(
        raw, direction=args.direction, split="train", model_family=args.model_family,
        rels=rels, langs=langs, layers=args.layers, heads=args.heads, arc_weight_agg=args.arc_weight_agg,
    )
    test_summary = main_summary
    train_test = comparison_table(train_summary, test_summary, "train", "test")
    save_csv(train_test, out_dir / "train_test_stability.csv", args.round, manifest, "Train vs. test stability of aggregate metrics.")

    # 3. Direction robustness: head_to_dep vs dep_to_head.
    alt_dir = "dep_to_head" if args.direction == "head_to_dep" else "head_to_dep"
    alt_summary, _ = compute_summary(
        raw, direction=alt_dir, split=args.split, model_family=args.model_family,
        rels=rels, langs=langs, layers=args.layers, heads=args.heads, arc_weight_agg=args.arc_weight_agg,
    )
    direction_tbl = comparison_table(main_summary, alt_summary, args.direction, alt_dir)
    save_csv(direction_tbl, out_dir / "direction_control.csv", args.round, manifest, "Main attention direction vs. alternative direction.")

    # 4. Monolingual vs. mBERT.
    mbert_summary, _ = compute_summary(
        raw, direction=args.direction, split=args.split, model_family="mbert",
        rels=rels, langs=langs, layers=args.layers, heads=args.heads, arc_weight_agg=args.arc_weight_agg,
    )
    mono_mbert = comparison_table(main_summary, mbert_summary, "mono", "mbert")
    save_csv(mono_mbert, out_dir / "mono_vs_mbert_control.csv", args.round, manifest, "Monolingual models vs. mBERT profiles.")

    # 5. With/without Portuguese--Galician pair at pairwise aggregation level.
    def not_pt_gl(a: str, b: str) -> bool:
        return set([a, b]) != {"pt", "gl"}

    all_pairs_agg = aggregate_pair_subset(main_pairs, rels, "all_pairs", lambda a, b: True)
    no_pt_gl_agg = aggregate_pair_subset(main_pairs, rels, "without_pt_gl_pair", not_pt_gl)
    ptgl_tbl = pd.concat([all_pairs_agg, no_pt_gl_agg], ignore_index=True)
    save_csv(ptgl_tbl, out_dir / "without_pt_gl_pair.csv", args.round, manifest, "Pairwise aggregate metrics with and without the Portuguese--Galician pair.")

    # 6. Romance--Romance vs. German--Romance control.
    def romance_romance_no_ptgl(a: str, b: str) -> bool:
        return (a in ROMANCE) and (b in ROMANCE) and set([a, b]) != {"pt", "gl"}

    def german_romance(a: str, b: str) -> bool:
        return (a == GERMAN and b in ROMANCE) or (b == GERMAN and a in ROMANCE)

    rr = aggregate_pair_subset(main_pairs, rels, "romance_romance_excluding_pt_gl", romance_romance_no_ptgl)
    gr = aggregate_pair_subset(main_pairs, rels, "german_romance", german_romance)
    rg = pd.concat([rr, gr], ignore_index=True)
    save_csv(rg, out_dir / "romance_vs_german_control.csv", args.round, manifest, "Romance--Romance excluding PT--GL vs. German--Romance pair groups.")

    # 7. Optional two Galician model control.
    gl_path_a = Path(args.gl_model_a_csv) if args.gl_model_a_csv else None
    gl_path_b = Path(args.gl_model_b_csv) if args.gl_model_b_csv else None
    if gl_path_a and gl_path_b and gl_path_a.exists() and gl_path_b.exists():
        gl_tbl = make_same_language_control(
            str(gl_path_a), str(gl_path_b), args.gl_model_a_label, args.gl_model_b_label,
            args.direction, args.split, rels, args.layers, args.heads, args.arc_weight_agg,
        )
        save_csv(gl_tbl, out_dir / "galician_two_models_control.csv", args.round, manifest, "Two independently trained Galician BERT models.")
    else:
        missing_note = pd.DataFrame([{
            "control": "galician_two_models",
            "status": "not_computed",
            "reason": "gl_model_a_csv and/or gl_model_b_csv were not provided or do not exist",
            "gl_model_a_csv": str(args.gl_model_a_csv),
            "gl_model_b_csv": str(args.gl_model_b_csv),
        }])
        save_csv(missing_note, out_dir / "galician_two_models_control.csv", args.round, manifest, "Two-Galician-model control not computed because auxiliary files were unavailable.")

    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(out_dir / "controls_manifest.csv", index=False, encoding="utf-8")

    print("[OK] controls written to:", out_dir.resolve())
    print(manifest_df.to_string(index=False))


if __name__ == "__main__":
    main()
