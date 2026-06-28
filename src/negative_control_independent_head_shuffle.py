#!/usr/bin/env python3
"""
Negative control: independently shuffle heads within each relation.

GPU note:
  This script is GPU-aware, but the core operations are based on pandas/numpy/scipy.
  The Hungarian algorithm from scipy.optimize.linear_sum_assignment runs on CPU.
  GPU is useful for Transformer inference, not for this small aggregated-control step.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from negative_control_shuffle_deprel import (
    DEPS,
    build_tensors,
    compute_pairwise_metrics,
    summarize_pairwise,
)


def report_device() -> None:
    try:
        import torch
        print("[DEVICE] torch:", torch.__version__)
        print("[DEVICE] cuda available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("[DEVICE] cuda device:", torch.cuda.get_device_name(0))
        else:
            print("[DEVICE] running on CPU")
    except Exception as e:
        print("[DEVICE] torch unavailable; running on CPU")
        print("[DEVICE] reason:", e)


def shuffle_heads_independently(
    tensors: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """
    T shape: [dep, layer, head]

    For each language, dependency and layer, shuffle head positions independently.
    This breaks cross-relation head identity while preserving values.
    """
    out = {}

    for lang, T in tensors.items():
        S = T.copy()

        n_dep, n_layer, n_head = S.shape

        for d in range(n_dep):
            for l in range(n_layer):
                perm = rng.permutation(n_head)
                S[d, l, :] = S[d, l, perm]

        out[lang] = S

    return out


def percentile_ci(x: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/attention_all_splits.csv")
    ap.add_argument("--output", default="revision_tables/negative_control_independent_head_shuffle_summary.csv")
    ap.add_argument("--split", default="test")
    ap.add_argument("--direction", default="head_to_dep")
    ap.add_argument("--model_family", default="mono")
    ap.add_argument("--n_iter", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260627)
    args = ap.parse_args()

    report_device()

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
        shuf_tensors = shuffle_heads_independently(tensors, rng)
        shuf_pairwise = compute_pairwise_metrics(shuf_tensors, counts, DEPS)
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
        null_mean = float(np.mean(null_vals))

        p_null_ge_real = float((1 + np.sum(null_vals >= real_val)) / (len(null_vals) + 1))

        rows.append({
            "deprel": dep,
            "real_rho_micro_index": float(r["rho_micro_index"]),
            "real_rho_macro": float(r["rho_macro"]),
            "real_rho_micro_matched_loo": real_val,
            "null_mean_rho_micro_matched_loo": null_mean,
            "null_ci_low": null_lo,
            "null_ci_high": null_hi,
            "real_minus_null_mean": real_val - null_mean,
            "p_null_ge_real": p_null_ge_real,
            "n_null_iterations": len(null_vals),
        })

    out = pd.DataFrame(rows)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    null_df.to_csv(
        Path(args.output).with_name("negative_control_independent_head_shuffle_null_iterations.csv"),
        index=False,
    )

    print("[OK] wrote", args.output)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
