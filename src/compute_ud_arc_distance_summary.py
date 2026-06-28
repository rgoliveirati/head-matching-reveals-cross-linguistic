#!/usr/bin/env python3
"""
Compute UD arc-distance summaries for the relations used in the paper.

Outputs:
  revision_tables/ud_arc_distance_by_relation_lang_split.csv
  revision_tables/ud_arc_distance_global_by_relation.csv
  revision_outputs/ud_arc_distance_arcs.csv

This script does not run any Transformer model. It only reads CoNLL-U files
and summarizes the linear distance of syntactic arcs.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEPS = {"nsubj", "obj", "case", "amod"}


LANG_PATTERNS = [
    ("pt", ["Portuguese", "pt_"]),
    ("gl", ["Galician", "gl_"]),
    ("es", ["Spanish", "es_"]),
    ("fr", ["French", "fr_"]),
    ("it", ["Italian", "it_"]),
    ("ro", ["Romanian", "ro_"]),
    ("de", ["German", "de_"]),
]


def infer_lang(path: Path) -> str:
    s = str(path)
    for lang, pats in LANG_PATTERNS:
        for pat in pats:
            if pat.lower() in s.lower():
                return lang
    return "unknown"


def infer_split(path: Path) -> str:
    name = path.name.lower()
    if "train" in name:
        return "train"
    if "dev" in name:
        return "dev"
    if "test" in name:
        return "test"
    return "unknown"


def infer_treebank(path: Path) -> str:
    parts = list(path.parts)
    for p in parts:
        if p.startswith("UD_"):
            return p
    # fallback: parent directory
    return path.parent.name


def is_int_id(x: str) -> bool:
    return re.fullmatch(r"\d+", x) is not None


def crossing_arcs(arcs: list[tuple[int, int]]) -> set[int]:
    """
    Return indices of arcs that cross at least one other arc.
    This is a simple crossing-arc approximation for non-projectivity.
    Shared endpoints are ignored.
    """
    crossing = set()

    intervals = []
    for i, (h, d) in enumerate(arcs):
        if h == 0:
            continue
        a, b = sorted((h, d))
        if a == b:
            continue
        intervals.append((i, a, b, h, d))

    for idx1 in range(len(intervals)):
        i, a, b, h1, d1 = intervals[idx1]
        for idx2 in range(idx1 + 1, len(intervals)):
            j, c, d, h2, d2 = intervals[idx2]

            # Ignore shared endpoints
            if len({h1, d1, h2, d2}) < 4:
                continue

            if (a < c < b < d) or (c < a < d < b):
                crossing.add(i)
                crossing.add(j)

    return crossing


def parse_conllu(path: Path) -> list[dict]:
    rows = []

    sent_id = None
    sent_index = 0
    tokens = []

    def flush_sentence():
        nonlocal tokens, sent_id, sent_index, rows

        if not tokens:
            return

        sent_index += 1

        arcs = []
        token_records = []

        for tok in tokens:
            tid = tok["id"]
            head = tok["head"]
            if head == 0:
                continue
            arcs.append((head, tid))
            token_records.append(tok)

        crossing_idx = crossing_arcs(arcs)

        arc_pos = 0
        for tok in tokens:
            tid = tok["id"]
            head = tok["head"]
            dep = tok["deprel"]

            if head == 0:
                continue

            dist = abs(tid - head)

            is_crossing = arc_pos in crossing_idx
            arc_pos += 1

            if dep not in DEPS:
                continue

            rows.append({
                "file": str(path),
                "treebank": infer_treebank(path),
                "lang": infer_lang(path),
                "split": infer_split(path),
                "sent_id": sent_id if sent_id is not None else f"{path.name}:{sent_index}",
                "token_id": tid,
                "head_id": head,
                "deprel": dep,
                "distance": dist,
                "direction": "right" if tid > head else "left",
                "is_adjacent": dist == 1,
                "is_short_le_2": dist <= 2,
                "is_medium_3_5": 3 <= dist <= 5,
                "is_long_ge_6": dist >= 6,
                "crossing_arc": bool(is_crossing),
            })

        tokens = []
        sent_id = None

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")

            if not line:
                flush_sentence()
                continue

            if line.startswith("#"):
                if line.startswith("# sent_id"):
                    sent_id = line.split("=", 1)[-1].strip()
                continue

            cols = line.split("\t")
            if len(cols) != 10:
                continue

            if not is_int_id(cols[0]):
                # Skip multiword tokens and empty nodes
                continue

            if not is_int_id(cols[6]):
                continue

            tid = int(cols[0])
            head = int(cols[6])
            deprel = cols[7]

            # Strip subtypes, e.g. nsubj:pass -> nsubj
            deprel_base = deprel.split(":", 1)[0]

            tokens.append({
                "id": tid,
                "head": head,
                "deprel": deprel_base,
            })

    flush_sentence()
    return rows


def summarize(arcs: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []

    for key, sub in arcs.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)

        d = sub["distance"].to_numpy(dtype=float)

        row = dict(zip(group_cols, key))
        row.update({
            "n_arcs": int(len(sub)),
            "mean_distance": float(np.mean(d)),
            "median_distance": float(np.median(d)),
            "p90_distance": float(np.percentile(d, 90)),
            "max_distance": int(np.max(d)),
            "pct_adjacent_dist_1": float(np.mean(sub["is_adjacent"]) * 100),
            "pct_short_le_2": float(np.mean(sub["is_short_le_2"]) * 100),
            "pct_medium_3_5": float(np.mean(sub["is_medium_3_5"]) * 100),
            "pct_long_ge_6": float(np.mean(sub["is_long_ge_6"]) * 100),
            "pct_crossing_arc": float(np.mean(sub["crossing_arc"]) * 100),
            "pct_right_dependent": float(np.mean(sub["direction"] == "right") * 100),
        })
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out_dir", default="revision_tables")
    ap.add_argument("--arcs_out", default="revision_outputs/ud_arc_distance_arcs.csv")
    args = ap.parse_args()

    root = Path(args.root)
    files = sorted(list(root.rglob("*.conllu")) + list(root.rglob("*.conll")))

    # Avoid duplicated hidden notebook checkpoints or irrelevant generated files
    files = [p for p in files if ".ipynb_checkpoints" not in str(p)]

    if not files:
        raise SystemExit("No .conllu/.conll files found.")

    print("Found files:")
    for p in files:
        print("-", p)

    all_rows = []
    for p in files:
        all_rows.extend(parse_conllu(p))

    arcs = pd.DataFrame(all_rows)

    if arcs.empty:
        raise SystemExit("No target arcs found for nsubj/obj/case/amod.")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.arcs_out).parent.mkdir(parents=True, exist_ok=True)

    arcs.to_csv(args.arcs_out, index=False)

    by_lang_split = summarize(
        arcs,
        ["lang", "treebank", "split", "deprel"],
    ).sort_values(["deprel", "lang", "split"])

    global_by_rel = summarize(
        arcs,
        ["deprel"],
    ).sort_values("deprel")

    by_lang_split.to_csv(
        Path(args.out_dir) / "ud_arc_distance_by_relation_lang_split.csv",
        index=False,
    )
    global_by_rel.to_csv(
        Path(args.out_dir) / "ud_arc_distance_global_by_relation.csv",
        index=False,
    )

    print("\n[GLOBAL BY RELATION]")
    print(global_by_rel.round(3).to_string(index=False))

    print("\n[OK] wrote:")
    print("-", Path(args.out_dir) / "ud_arc_distance_by_relation_lang_split.csv")
    print("-", Path(args.out_dir) / "ud_arc_distance_global_by_relation.csv")
    print("-", args.arcs_out)


if __name__ == "__main__":
    main()
