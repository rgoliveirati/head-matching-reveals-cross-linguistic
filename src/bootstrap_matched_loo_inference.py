#!/usr/bin/env python3
"""
Bootstrap and paired sign-flip inference for permutation-aware head matching.

Input:
  results/controls/main_pairwise_head_matching.csv

Output:
  revision_tables/matched_loo_inference_by_relation.csv

The unit of resampling is the language pair.
This provides pair-level uncertainty for:
  - rho_micro_matched_loo
  - delta = rho_micro_matched_loo - rho_micro_index
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def pick_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"None of these columns found: {candidates}. Available: {list(df.columns)}")
    return None


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)

    if not np.any(mask):
        return float("nan")

    return float(np.sum(values[mask] * weights[mask]) / np.sum(weights[mask]))


def ci_percentile(x: np.ndarray, lo: float = 2.5, hi: float = 97.5) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return float("nan"), float("nan")

    return float(np.percentile(x, lo)), float(np.percentile(x, hi))


def infer_weights(df: pd.DataFrame) -> np.ndarray:
    direct = pick_col(df, ["weight", "w", "pair_weight"], required=False)
    if direct is not None:
        return df[direct].to_numpy(dtype=float)

    pairs = [
        ("n_arcs_a", "n_arcs_b"),
        ("arcs_a", "arcs_b"),
        ("count_a", "count_b"),
        ("n_a", "n_b"),
        ("n1", "n2"),
    ]

    for a, b in pairs:
        if a in df.columns and b in df.columns:
            return np.minimum(df[a].to_numpy(dtype=float), df[b].to_numpy(dtype=float))

    return np.ones(len(df), dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/controls/main_pairwise_head_matching.csv")
    ap.add_argument("--output", default="revision_tables/matched_loo_inference_by_relation.csv")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--n_perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260627)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(args.input)

    rel_col = pick_col(df, ["deprel", "dep", "relation", "rel"])
    idx_col = pick_col(df, ["rho_micro_index", "rho_micro_idx", "rho_idx"])
    loo_col = pick_col(df, ["rho_micro_matched_loo", "rho_matched_loo", "rho_micro_loo"])
    macro_col = pick_col(df, ["rho_macro", "rho_macro_weighted", "rho_macro_mean"], required=False)

    rows = []

    for rel, sub in df.groupby(rel_col, sort=False):
        sub = sub.copy().reset_index(drop=True)
        w = infer_weights(sub)

        idx = sub[idx_col].to_numpy(dtype=float)
        loo = sub[loo_col].to_numpy(dtype=float)
        delta = loo - idx

        if macro_col is not None:
            macro = sub[macro_col].to_numpy(dtype=float)
            rho_macro = weighted_mean(macro, w)
        else:
            rho_macro = float("nan")

        rho_idx = weighted_mean(idx, w)
        rho_loo = weighted_mean(loo, w)
        delta_obs = weighted_mean(delta, w)

        n = len(sub)

        boot_loo = []
        boot_delta = []

        for _ in range(args.n_boot):
            sample_idx = rng.integers(0, n, size=n)
            boot_loo.append(weighted_mean(loo[sample_idx], w[sample_idx]))
            boot_delta.append(weighted_mean(delta[sample_idx], w[sample_idx]))

        loo_lo, loo_hi = ci_percentile(np.array(boot_loo))
        delta_lo, delta_hi = ci_percentile(np.array(boot_delta))

        # Paired sign-flip test for delta > 0 and two-sided delta != 0
        perm_delta = []
        for _ in range(args.n_perm):
            signs = rng.choice(np.array([-1.0, 1.0]), size=n)
            perm_delta.append(weighted_mean(delta * signs, w))

        perm_delta = np.array(perm_delta, dtype=float)

        p_delta_greater = float((1 + np.sum(perm_delta >= delta_obs)) / (args.n_perm + 1))
        p_delta_two_sided = float((1 + np.sum(np.abs(perm_delta) >= abs(delta_obs))) / (args.n_perm + 1))

        # Sign-flip test for matched LOO > 0
        perm_loo = []
        for _ in range(args.n_perm):
            signs = rng.choice(np.array([-1.0, 1.0]), size=n)
            perm_loo.append(weighted_mean(loo * signs, w))

        perm_loo = np.array(perm_loo, dtype=float)
        p_loo_greater_zero = float((1 + np.sum(perm_loo >= rho_loo)) / (args.n_perm + 1))

        rows.append({
            "deprel": rel,
            "n_language_pairs": n,
            "rho_micro_index": rho_idx,
            "rho_macro": rho_macro,
            "rho_micro_matched_loo": rho_loo,
            "rho_micro_matched_loo_ci_low": loo_lo,
            "rho_micro_matched_loo_ci_high": loo_hi,
            "delta_loo_minus_index": delta_obs,
            "delta_ci_low": delta_lo,
            "delta_ci_high": delta_hi,
            "p_delta_greater_than_zero": p_delta_greater,
            "p_delta_two_sided": p_delta_two_sided,
            "p_matched_loo_greater_than_zero": p_loo_greater_zero,
        })

    out = pd.DataFrame(rows)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"[OK] wrote {args.output}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
