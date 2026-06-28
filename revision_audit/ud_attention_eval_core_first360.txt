# ud_attention_eval_core.py
# ============================================================
# Núcleo de avaliação: UD (.conllu) × Atenção (Transformers HF)
# ============================================================

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

@dataclass
class UDToken:
    id: int
    form: str
    head: int
    deprel: str

@dataclass
class UDSentence:
    text: str
    spans: Dict[int, Tuple[int, int]]
    tokens: List[UDToken]

def _is_int_id(x: str) -> bool:
    return x.isdigit()

def parse_conllu_sentences(path: str, max_sents: Optional[int] = None) -> List[UDSentence]:
    sents: List[UDSentence] = []
    buf: List[str] = []

    def flush(buf_lines: List[str]):
        ud_tokens: List[UDToken] = []
        for line in buf_lines:
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 8:
                continue
            tid = cols[0]
            if not _is_int_id(tid):
                continue
            wid = int(tid)
            form = cols[1]
            head = cols[6]
            deprel = cols[7]
            if not head.isdigit():
                continue
            ud_tokens.append(UDToken(id=wid, form=form, head=int(head), deprel=deprel))

        if not ud_tokens:
            return

        parts: List[str] = []
        spans: Dict[int, Tuple[int, int]] = {}
        cur = 0
        for t in ud_tokens:
            if parts:
                parts.append(" ")
                cur += 1
            start = cur
            parts.append(t.form)
            cur += len(t.form)
            end = cur
            spans[t.id] = (start, end)

        text = "".join(parts)
        sents.append(UDSentence(text=text, spans=spans, tokens=ud_tokens))

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "":
                if buf:
                    flush(buf)
                    buf = []
                    if max_sents is not None and len(sents) >= max_sents:
                        break
            else:
                buf.append(line)
        if buf and (max_sents is None or len(sents) < max_sents):
            flush(buf)

    return sents

def map_words_to_token_spans(tokenizer, text: str, word_spans: Dict[int, Tuple[int, int]], max_len: int):
    enc = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
        return_offsets_mapping=True,
        truncation=True,
        max_length=int(max_len),
    )
    offsets = enc["offset_mapping"][0].tolist()

    word2tok: Dict[int, Tuple[int, int]] = {}
    for wid, (ws, we) in word_spans.items():
        idxs: List[int] = []
        for ti, (ts, te) in enumerate(offsets):
            if ts == 0 and te == 0:
                continue
            if not (te <= ws or ts >= we):
                idxs.append(ti)
        if idxs:
            word2tok[wid] = (min(idxs), max(idxs) + 1)
    return enc, word2tok

def mean_attention_between_spans(attn: np.ndarray, span_a: Tuple[int, int], span_b: Tuple[int, int]) -> float:
    a0, a1 = span_a
    b0, b1 = span_b
    if a0 >= a1 or b0 >= b1:
        return float("nan")
    sub = attn[a0:a1, b0:b1]
    if sub.size == 0:
        return float("nan")
    return float(sub.mean())
