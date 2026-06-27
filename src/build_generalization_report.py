# build_generalization_report.py
# ============================================================

from __future__ import annotations
import argparse
import datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd

def macro_micro_reading(rho_layer: float, rho_head: float) -> str:
    if pd.isna(rho_layer) or pd.isna(rho_head):
        return "indeterminado"
    if (rho_layer >= 0.30) and (rho_head >= 0.25):
        return "macro_e_micro"
    if (rho_layer >= 0.30) and (rho_head < 0.20):
        return "macro_por_camada"
    if (rho_layer < 0.30) and (rho_head >= 0.25):
        return "micro_sem_macro"
    return "fraca/ausente"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_csv", type=str, required=True)
    ap.add_argument("--entropy_csv", type=str, required=True)
    ap.add_argument("--out_md", type=str, default="achados_generalizacao_romanic.md")
    ap.add_argument("--out_html", type=str, default="achados_generalizacao_romanic.html")
    args = ap.parse_args()

    metrics = pd.read_csv(args.metrics_csv)
    entropy = pd.read_csv(args.entropy_csv)

    for c in ["stability_score_0_1","spearman_mean_layer","spearman_wmean","entropy_norm_mean","entropy_norm_std"]:
        if c in metrics.columns:
            metrics[c] = pd.to_numeric(metrics[c], errors="coerce")
    if "entropy_norm" in entropy.columns:
        entropy["entropy_norm"] = pd.to_numeric(entropy["entropy_norm"], errors="coerce")

    if "split" not in metrics.columns:
        metrics["split"] = "(todos)"
    if "split" not in entropy.columns:
        entropy["split"] = "(todos)"

    metrics["macro_micro_reading"] = [
        macro_micro_reading(a, b) for a, b in zip(metrics["spearman_mean_layer"], metrics["spearman_wmean"])
    ]

    lines = []
    lines.append("# Relatório automático — generalização atencional (UD × atenção)\n")
    lines.append(f"*Gerado em:* {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    lines.append("## 0) Premissa (por que split importa)\n")
    lines.append(
        "- **Train**: descoberta (gera hipóteses).\n"
        "- **Dev**: calibração (quando existir).\n"
        "- **Test**: confirmação (evidência mais forte).\n"
        "Se o treebank não tiver dev, o pipeline marca `train_fallback_dev`.\n"
    )

    for sp in sorted(metrics["split"].dropna().unique().tolist()):
        sub = metrics[metrics["split"] == sp].sort_values("stability_score_0_1", ascending=False)
        lines.append(f"\n## 1) Métricas por relação — split = `{sp}`\n")
        keep = [c for c in [
            "deprel","langs_used","spearman_mean_layer","spearman_wmean",
            "entropy_norm_mean","entropy_norm_std","stability_score_0_1","generalization_type","macro_micro_reading"
        ] if c in sub.columns]
        lines.append(sub[keep].to_markdown(index=False))

    lines.append("\n\n## 2) Entropia por língua (pivot)\n")
    for sp in sorted(entropy["split"].dropna().unique().tolist()):
        e = entropy[entropy["split"] == sp]
        lines.append(f"\n### split = `{sp}`\n")
        try:
            piv = e.pivot_table(index="lang", columns="deprel", values="entropy_norm", aggfunc="mean")
            lines.append(piv.round(3).to_markdown())
        except Exception:
            lines.append("(pivot indisponível)")

    md_text = "\n".join(lines)
    out_md = Path(args.out_md)
    out_md.write_text(md_text, encoding="utf-8")

    out_html = Path(args.out_html)
    try:
        import markdown as mdlib
        body = mdlib.markdown(md_text, extensions=["tables", "fenced_code", "toc"])
    except Exception:
        body = "<pre>" + md_text.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;") + "</pre>"

    html = f"""<!doctype html>
<html lang="pt-br">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Achados — Generalização Atencional</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; margin: 24px; max-width: 1100px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 18px 0; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
    th {{ background: #f5f5f5; }}
    tr:nth-child(even) td {{ background: #fcfcfc; }}
    code, pre {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }}
  </style>
</head>
<body>
{body}
</body>
</html>"""
    out_html.write_text(html, encoding="utf-8")

    print("[OK] md:", out_md.resolve())
    print("[OK] html:", out_html.resolve())

if __name__ == "__main__":
    main()
