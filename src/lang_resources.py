"""
lang_resources.py (v2) — Multi-língua (UD + modelos HF)

Requisitos atendidos:
1) Hub: selecionar par (modelo recomendado da língua OU mBERT) + treebank UD correspondente.
2) Páginas: sempre oferecer a lista "global" (união) de modelos BERT/Roberta configurados no projeto.
3) Download UD:
   - ZIPs em ./ud_cache
   - Arquivos .conllu em ./ud_treebanks/<repo>/
   - Copia TODOS os .conllu encontrados no repositório UD (não só train/dev/test)

Observação:
- Alguns treebanks não têm split dev (ex.: gl_treegal). O código detecta automaticamente.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests


# ============================================================
# 1) Catálogo UD (língua -> repo + prefixo)
# ============================================================
# prefixo: "<lang>_<treebank>-ud" (ex.: pt_bosque-ud)
UD_TREEBANKS: Dict[str, Dict[str, str]] = {
    "pt": {"repo": "UD_Portuguese-Bosque", "prefix": "pt_bosque-ud"},
    "gl": {"repo": "UD_Galician-TreeGal", "prefix": "gl_treegal-ud"},
    "es": {"repo": "UD_Spanish-AnCora", "prefix": "es_ancora-ud"},
    "it": {"repo": "UD_Italian-ISDT", "prefix": "it_isdt-ud"},
    "fr": {"repo": "UD_French-GSD", "prefix": "fr_gsd-ud"},
    "ro": {"repo": "UD_Romanian-RRT", "prefix": "ro_rrt-ud"},
    "de": {"repo": "UD_German-GSD",   "prefix": "de_gsd-ud"},
}

# ============================================================
# 2) Catálogo de modelos por língua (recomendado + mBERT)
# ============================================================
# Isto é um "starter pack" prático. As páginas expõem a lista global (união).
HF_MODELS: Dict[str, Dict[str, object]] = {
    "pt": {
        "recommended": "neuralmind/bert-base-portuguese-cased",
        "extras": [
            "neuralmind/bert-large-portuguese-cased",
        ],
    },
    "gl": {
        "recommended": "marcosgg/bert-base-gl-cased",
        "extras": [
            "fpuentes/bert-galician",
        ],
    },
    "es": {
        "recommended": "dccuchile/bert-base-spanish-wwm-cased",
        "extras": [],
    },
    "it": {
        "recommended": "dbmdz/bert-base-italian-xxl-cased",
        "extras": [],
    },
    "fr": {
        # CamemBERT é RoBERTa; funciona no seu pipeline se você usar AutoTokenizer/AutoModel.
        "recommended": "camembert-base",
        "extras": [],
    },
    "ro": {
        "recommended": "dumitrescustefan/bert-base-romanian-cased-v1",
        "extras": [],
    },
    "de": {
    "recommended": "bert-base-german-cased",
    "extras": []
},
}

MBERT = "bert-base-multilingual-cased"
XLMR = "xlm-roberta-base"


# ============================================================
# 3) Pastas do projeto
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
UD_CACHE_DIR = PROJECT_ROOT / "ud_cache"         # ZIPs
UD_TREEBANK_DIR = PROJECT_ROOT / "ud_treebanks"  # .conllu extraídos


# ============================================================
# 4) Download e extração UD
# ============================================================
def _ud_zip_urls(repo: str) -> Tuple[str, str]:
    base = f"https://github.com/UniversalDependencies/{repo}/archive/refs/heads"
    return (f"{base}/main.zip", f"{base}/master.zip")


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def ensure_ud_treebank(lang: str, variant: Optional[str] = None) -> Dict[str, object]:
    """
    Garante que TODOS os .conllu do treebank UD estejam em:
      ./ud_treebanks/<repo>/

    ZIP fica em:
      ./ud_cache/<repo>.zip

    Retorna caminhos normalizados (train/dev/test se existirem) + lista de todos os conllu.

    {
      "repo": str,
      "prefix": str,
      "treebank_dir": Path,
      "all_conllu": List[Path],
      "train": Path,
      "dev": Optional[Path],
      "test": Path,
      "has_dev": bool,
    }
    """
    if lang not in UD_TREEBANKS:
        raise ValueError(f"Língua não suportada: {lang}. Suportadas: {sorted(UD_TREEBANKS)}")

    repo = UD_TREEBANKS[lang]["repo"]
    prefix = UD_TREEBANKS[lang]["prefix"]

    treebank_dir = UD_TREEBANK_DIR / (variant or repo)
    train_path = treebank_dir / f"{prefix}-train.conllu"
    dev_path   = treebank_dir / f"{prefix}-dev.conllu"
    test_path  = treebank_dir / f"{prefix}-test.conllu"

    # Se já existe qualquer conllu, assumimos preparado.
    if treebank_dir.exists() and any(treebank_dir.glob("*.conllu")):
        has_dev = dev_path.exists()
        # train/test são esperados; se faltarem, força re-download
        if train_path.exists() and test_path.exists():
            all_conllu = sorted(treebank_dir.glob("*.conllu"))
            return {
                "repo": repo,
                "prefix": prefix,
                "treebank_dir": treebank_dir,
                "all_conllu": all_conllu,
                "train": train_path,
                "dev": dev_path if has_dev else None,
                "test": test_path,
                "has_dev": has_dev,
            }

    UD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    UD_TREEBANK_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = UD_CACHE_DIR / f"{repo}.zip"
    last_err = None

    for url in _ud_zip_urls(repo):
        try:
            _download(url, zip_path)

            # extrai em pasta temporária (dentro de ud_cache)
            extract_dir = UD_CACHE_DIR / f"_{repo}_extract"
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
            extract_dir.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(extract_dir)

            # coleta TODOS os .conllu do repo
            conllu_files = list(extract_dir.rglob("*.conllu"))
            if not conllu_files:
                raise RuntimeError(f"Nenhum .conllu encontrado no zip de {repo} (url={url})")

            # cria destino e limpa versões antigas
            if treebank_dir.exists():
                shutil.rmtree(treebank_dir)
            treebank_dir.mkdir(parents=True, exist_ok=True)

            # copia todos os conllu para a pasta final (achatando)
            for p in conllu_files:
                dest = treebank_dir / p.name
                shutil.copy2(p, dest)

            # valida splits básicos
            if not train_path.exists():
                raise FileNotFoundError(f"Não encontrei {train_path.name} no treebank extraído de {repo}.")
            if not test_path.exists():
                raise FileNotFoundError(f"Não encontrei {test_path.name} no treebank extraído de {repo}.")

            has_dev = dev_path.exists()

            # limpa extração temporária (mantém zip cache)
            try:
                shutil.rmtree(extract_dir)
            except Exception:
                pass

            all_conllu = sorted(treebank_dir.glob("*.conllu"))
            return {
                "repo": repo,
                "prefix": prefix,
                "treebank_dir": treebank_dir,
                "all_conllu": all_conllu,
                "train": train_path,
                "dev": dev_path if has_dev else None,
                "test": test_path,
                "has_dev": has_dev,
            }

        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"Falha ao preparar UD para {lang} ({repo}). Último erro: {last_err}")


# ============================================================
# 5) Modelos — lista global (união) p/ páginas
# ============================================================
def all_model_options() -> List[str]:
    """
    Lista global de modelos (união do catálogo do projeto).
    Mantém ordem estável: recomendados por língua, extras por língua, depois mBERT/xlm-r.
    """
    out: List[str] = []
    # recomendados
    for lang, meta in HF_MODELS.items():
        rec = str(meta.get("recommended"))
        if rec and rec not in out:
            out.append(rec)
    # extras
    for lang, meta in HF_MODELS.items():
        for m in (meta.get("extras") or []):
            m = str(m)
            if m and m not in out:
                out.append(m)
    # baselines
    for m in [MBERT, XLMR]:
        if m not in out:
            out.append(m)
    return out


def recommended_model_for(lang: str) -> str:
    return str(HF_MODELS.get(lang, {}).get("recommended") or MBERT)


# ============================================================
# 6) UI helpers (Streamlit)
# ============================================================
def sidebar_hub_pair(st) -> Dict[str, object]:
    """
    Sidebar do HUB:
      - escolhe língua
      - escolhe "Modelo: recomendado" OU "mBERT"
      - prepara UD (com botão)
    Armazena em session_state:
      - lang
      - model_id
      - model_mode ("recommended" | "mbert")
      - ud_split (train/dev/test de acordo com o que existe)
      - ud_treebank_dir
    """
    langs = list(UD_TREEBANKS.keys())
    lang_default = st.session_state.get("lang") or "pt"
    if lang_default not in langs:
        lang_default = langs[0]

    lang = st.selectbox("Língua", langs, index=langs.index(lang_default))
    st.session_state["lang"] = lang

    mode_default = st.session_state.get("model_mode") or "recommended"
    mode = st.radio("Modelo", ["recommended", "mbert"], index=0 if mode_default == "recommended" else 1,
                    format_func=lambda x: "Modelo da língua (recomendado)" if x == "recommended" else "mBERT (baseline)")
    st.session_state["model_mode"] = mode

    model_id = recommended_model_for(lang) if mode == "recommended" else MBERT
    st.session_state["model_id"] = model_id

    st.caption(f"Selecionado: {model_id}")

    # Split UD: determinado depois que o UD for baixado. Default "train".
    st.session_state["ud_split"] = st.session_state.get("ud_split") or "train"

    return {"lang": lang, "model_mode": mode, "model_id": model_id}


def sidebar_pages_all_models(st) -> Dict[str, object]:
    """
    Sidebar das PÁGINAS:
      - mostra todos os modelos do catálogo do projeto (união)
      - permite custom model id
      - permite preparar UD se ainda não estiver pronto
    """
    st.subheader("Configuração (página)")
    lang = st.session_state.get("lang") or "pt"
    st.caption(f"Língua atual (do hub): {lang}")

    opts = all_model_options()
    current = st.session_state.get("model_id") or recommended_model_for(lang)
    if current not in opts:
        opts = [current] + opts

    model_id = st.selectbox("Modelo (lista global)", opts, index=opts.index(current))
    custom = st.text_input("Ou informe um model_id (opcional)", value="")
    if (custom or "").strip():
        model_id = (custom or "").strip()

    st.session_state["model_id"] = model_id

    # UD split só oferece o que existe; se UD ainda não existe, deixa padrão.
    ud_split = st.session_state.get("ud_split") or "train"
    st.session_state["ud_split"] = ud_split

    return {"lang": lang, "model_id": model_id, "ud_split": ud_split}
