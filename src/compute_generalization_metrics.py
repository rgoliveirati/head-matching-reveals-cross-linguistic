# compute_generalization_metrics.py
# ============================================================

from __future__ import annotations
import argparse
from itertools import combinations
from math import log2
from pathlib import Path
import numpy as np
import pandas as pd

def pearson_corr(a, b):
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])

def spearman_corr(a, b):
    ra = pd.Series(a).rank(method="average").to_numpy()
    rb = pd.Series(b).rank(method="average").to_numpy()
    return pearson_corr(ra, rb)

def shannon_entropy(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    if p.size == 0:
        return np.nan
    return float(-(p * np.log2(p)).sum())

def clamp01(x):
    return float(np.clip(x, 0.0, 1.0))

def classify_macro_micro(rho_layer, rho_head, thr_layer=0.30, thr_head=0.20):
    if pd.isna(rho_layer) or pd.isna(rho_head):
        return "insuficiente"
    if rho_layer >= thr_layer and rho_head < thr_head:
        return "macro_por_camada"
    if rho_layer >= thr_layer and rho_head >= thr_head:
        return "macro_e_micro"
    if rho_layer < thr_layer and rho_head >= thr_head:
        return "micro_sem_macro (suspeito)"
    return "fraca/ausente"

def build_index_pairs(L: int, H: int):
    pairs = [(l, h) for l in range(1, L + 1) for h in range(1, H + 1)]
    return pairs, {p: i for i, p in enumerate(pairs)}

def build_vector(sub: pd.DataFrame, index_pairs, pair_to_idx):
    v = np.zeros(len(index_pairs), dtype=float)
    for _, r in sub.iterrows():
        i = pair_to_idx.get((int(r["layer"]), int(r["head"])))
        if i is not None:
            v[i] = float(r["mean_attention"])
    return v

def build_layer_vector(sub: pd.DataFrame, L: int):
    out = np.zeros(L, dtype=float)
    for layer in range(1, L + 1):
        x = sub[sub["layer"] == layer]["mean_attention"].to_numpy(dtype=float)
        out[layer - 1] = float(np.mean(x)) if x.size else 0.0
    return out

def compute_for_subset(df: pd.DataFrame, w_entropy_std: float = 0.10,
                       w_full: float = 0.35, w_layer: float = 0.45, w_conc: float = 0.10, w_cons: float = 0.10):
    norm = w_full + w_layer + w_conc + w_cons
    w_full, w_layer, w_conc, w_cons = [x / norm for x in (w_full, w_layer, w_conc, w_cons)]

    L = int(df["layer"].max())
    H = int(df["head"].max())
    index_pairs, pair_to_idx = build_index_pairs(L, H)

    langs = sorted(df["lang"].unique().tolist())
    deprels = sorted(df["deprel"].unique().tolist())

    vectors = {}
    weights = {}
    for dep in deprels:
        for lang in langs:
            sub = df[(df["deprel"] == dep) & (df["lang"] == lang)]
            if sub.empty:
                continue
            vectors[(lang, dep)] = build_vector(sub, index_pairs, pair_to_idx)
            weights[(lang, dep)] = float(sub["n_arcs"].sum())

    rows_full = []
    for dep in deprels:
        available = [lang for lang in langs if (lang, dep) in vectors]
        if len(available) < 2:
            continue
        cors_s_w = []
        w_sum = 0.0
        for l1, l2 in combinations(available, 2):
            cs = spearman_corr(vectors[(l1, dep)], vectors[(l2, dep)])
            w = min(weights[(l1, dep)], weights[(l2, dep)])
            if not np.isnan(cs):
                cors_s_w.append(cs * w)
            w_sum += w
        rows_full.append({"deprel": dep, "langs_used": len(available),
                          "spearman_wmean": (np.nansum(cors_s_w) / w_sum) if w_sum > 0 else np.nan})
    full_corr_df = pd.DataFrame(rows_full, columns=["deprel","langs_used","spearman_wmean"])

    layer_vectors = {}
    for dep in deprels:
        for lang in langs:
            sub = df[(df["deprel"] == dep) & (df["lang"] == lang)]
            if sub.empty:
                continue
            layer_vectors[(lang, dep)] = build_layer_vector(sub, L=L)

    rows_layer = []
    for dep in deprels:
        available = [lang for lang in langs if (lang, dep) in layer_vectors]
        if len(available) < 2:
            continue
        cors_s = []
        for l1, l2 in combinations(available, 2):
            cors_s.append(spearman_corr(layer_vectors[(l1, dep)], layer_vectors[(l2, dep)]))
        rows_layer.append({"deprel": dep, "langs_used": len(available), "spearman_mean_layer": np.nanmean(cors_s)})
    layer_corr_df = pd.DataFrame(rows_layer, columns=["deprel","langs_used","spearman_mean_layer"])

    # Robustez: se não há pares suficientes em algum dos componentes (micro/macro),
    # o DataFrame pode ficar vazio (sem colunas) e merges quebram.
    # Nesse caso, retornamos métricas vazias para este subset.
    if full_corr_df.empty or layer_corr_df.empty:
        empty_metrics = pd.DataFrame(columns=[
            "deprel","langs_used","spearman_wmean","spearman_mean_layer",
            "entropy_norm_mean","entropy_norm_std","stability_score_0_1","generalization_type"
        ])
        empty_entropy = pd.DataFrame(columns=["lang","deprel","entropy_norm"])
        return empty_metrics, empty_entropy

    entropy_rows = []
    for dep in deprels:
        for lang in langs:
            sub = df[(df["deprel"] == dep) & (df["lang"] == lang)]
            if sub.empty:
                continue
            v = build_vector(sub, index_pairs, pair_to_idx).clip(min=0)
            total = float(v.sum())
            if total <= 0:
                Hnorm = np.nan
            else:
                p = v / total
                Hbits = shannon_entropy(p)
                Hnorm = float(Hbits / log2(len(index_pairs)))
            entropy_rows.append({"lang": lang, "deprel": dep, "entropy_norm": Hnorm})
    entropy_df = pd.DataFrame(entropy_rows)
    
    if entropy_df.empty:
        entropy_summary = pd.DataFrame(columns=["deprel","entropy_norm_mean","entropy_norm_std"])
    else:
        entropy_summary = entropy_df.groupby("deprel", as_index=False).agg(
            entropy_norm_mean=("entropy_norm", "mean"),
            entropy_norm_std=("entropy_norm", "std"),
        )
    stab = (full_corr_df.merge(layer_corr_df, on="deprel", how="inner")
                      .merge(entropy_summary, on="deprel", how="inner"))

    # Harmoniza langs_used caso o merge tenha gerado duplicatas (ex.: langs_used_x, langs_used_y)
    if "langs_used" not in stab.columns:
        if "langs_used_x" in stab.columns and "langs_used_y" in stab.columns:
            stab["langs_used"] = stab[["langs_used_x","langs_used_y"]].min(axis=1).astype(int)
        elif "langs_used_x" in stab.columns:
            stab["langs_used"] = stab["langs_used_x"].astype(int)
        elif "langs_used_y" in stab.columns:
            stab["langs_used"] = stab["langs_used_y"].astype(int)

    if stab.empty:
        empty_metrics = pd.DataFrame(columns=[
            "deprel","langs_used","spearman_wmean","spearman_mean_layer",
            "entropy_norm_mean","entropy_norm_std","stability_score_0_1","generalization_type"
        ])
        return empty_metrics, entropy_df

    out_rows = []
    for _, r in stab.iterrows():
        corr_full = float(r["spearman_wmean"])
        corr_layer = float(r["spearman_mean_layer"])
        ent_mean = float(r["entropy_norm_mean"])
        ent_std = float(r["entropy_norm_std"]) if not pd.isna(r["entropy_norm_std"]) else 0.0

        s_full = clamp01((corr_full + 1) / 2)
        s_layer = clamp01((corr_layer + 1) / 2)
        s_conc = clamp01(1 - ent_mean)
        s_cons = clamp01(1 - min(ent_std / float(w_entropy_std), 1.0))

        score = w_full * s_full + w_layer * s_layer + s_conc * w_conc + s_cons * w_cons

        out_rows.append({
            "deprel": r["deprel"],
            "langs_used": int(r["langs_used"]),
            "spearman_wmean": corr_full,
            "spearman_mean_layer": corr_layer,
            "entropy_norm_mean": ent_mean,
            "entropy_norm_std": float(ent_std),
            "stability_score_0_1": float(score),
            "generalization_type": classify_macro_micro(corr_layer, corr_full),
        })

    return pd.DataFrame(out_rows).sort_values("stability_score_0_1", ascending=False), entropy_df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", type=str, required=True)
    ap.add_argument("--direction", type=str, default="head_to_dep")
    ap.add_argument("--splits", type=str, default="(todos)")
    ap.add_argument("--rels", type=str, default="(todos)")
    ap.add_argument("--model_family", type=str, default="(todos)")
    ap.add_argument("--out_dir", type=str, default=".")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    req = {"lang","deprel","layer","head","direction","mean_attention","n_arcs"}
    miss = req - set(df.columns)
    if miss:
        raise SystemExit(f"CSV inválido. Faltam colunas: {sorted(miss)}")

    df = df[df["direction"] == args.direction].copy()
    if args.splits != "(todos)" and "split" in df.columns:
        splits = [x.strip() for x in args.splits.split(",") if x.strip()]
        df = df[df["split"].isin(splits)].copy()
    if args.rels != "(todos)":
        rels = [x.strip() for x in args.rels.split(",") if x.strip()]
        df = df[df["deprel"].isin(rels)].copy()
    if args.model_family != "(todos)" and "model_family" in df.columns:
        fams = [x.strip() for x in args.model_family.split(",") if x.strip()]
        df = df[df["model_family"].isin(fams)].copy()

    if df.empty:
        raise SystemExit("DataFrame vazio após filtros.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_all = []
    entropy_all = []

    if "split" in df.columns:
        for sp in sorted(df["split"].dropna().unique().tolist()):
            dsub = df[df["split"] == sp].copy()
            m, e = compute_for_subset(dsub)
            m["split"] = sp
            e["split"] = sp
            metrics_all.append(m)
            entropy_all.append(e)

    m0, e0 = compute_for_subset(df)
    m0["split"] = "(todos)"
    e0["split"] = "(todos)"
    metrics_all.append(m0)
    entropy_all.append(e0)

    metrics_df = pd.concat(metrics_all, ignore_index=True)
    entropy_df = pd.concat(entropy_all, ignore_index=True)

    out_m = out_dir / "generalization_metrics_by_deprel.csv"
    out_e = out_dir / "entropy_by_lang.csv"
    metrics_df.to_csv(out_m, index=False, encoding="utf-8")
    entropy_df.to_csv(out_e, index=False, encoding="utf-8")

    print("[OK] métricas:", out_m.resolve())
    print("[OK] entropia:", out_e.resolve())

if __name__ == "__main__":
    main()
