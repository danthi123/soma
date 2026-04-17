"""Token-level F1, exact match, and ROUGE metrics for LongMemEval."""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass, field


def _normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> list[str]:
    return _normalize(text).split()


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if _normalize(prediction) == _normalize(reference) else 0.0


def _rouge_n_score(
    prediction: str,
    reference: str,
    n: int,
) -> float:
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if len(ref_tokens) < n:
        return 1.0 if pred_tokens == ref_tokens else 0.0
    if len(pred_tokens) < n:
        return 0.0

    def _ngrams(tokens: list[str], ng: int) -> Counter[tuple[str, ...]]:
        return Counter(tuple(tokens[i : i + ng]) for i in range(len(tokens) - ng + 1))

    pred_ng = _ngrams(pred_tokens, n)
    ref_ng = _ngrams(ref_tokens, n)
    overlap = sum((pred_ng & ref_ng).values())
    if overlap == 0:
        return 0.0
    precision = overlap / max(sum(pred_ng.values()), 1)
    recall = overlap / max(sum(ref_ng.values()), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: list[str], b: list[str]) -> int:
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        cur = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    return prev[n]


def rouge_l(prediction: str, reference: str) -> float:
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


@dataclass
class LongMemEvalScores:
    """Aggregated scores for a LongMemEval run."""

    f1: float = 0.0
    em: float = 0.0
    rouge1: float = 0.0
    rouge2: float = 0.0
    rougeL: float = 0.0
    per_type: dict[str, dict[str, float]] = field(default_factory=dict)
    n_total: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "f1": round(self.f1, 4),
            "em": round(self.em, 4),
            "rouge1": round(self.rouge1, 4),
            "rouge2": round(self.rouge2, 4),
            "rougeL": round(self.rougeL, 4),
            "per_type": self.per_type,
            "n_total": self.n_total,
        }


def compute_scores(
    predictions: list[dict[str, str]],
    references: list[dict[str, str]],
) -> LongMemEvalScores:
    """Compute aggregate metrics over matched prediction/reference pairs.

    Each dict must have ``question_id``, ``answer`` (reference) or
    ``hypothesis`` (prediction), and ``question_type``.
    """
    ref_map: dict[str, dict[str, str]] = {r["question_id"]: r for r in references}

    type_scores: dict[str, dict[str, list[float]]] = {}
    all_f1: list[float] = []
    all_em: list[float] = []
    all_r1: list[float] = []
    all_r2: list[float] = []
    all_rl: list[float] = []

    for pred in predictions:
        qid = pred["question_id"]
        ref = ref_map.get(qid)
        if ref is None:
            continue
        hyp = pred.get("hypothesis", "")
        gold = ref.get("answer", "")
        qtype = ref.get("question_type", "unknown")

        f1_val = token_f1(hyp, gold)
        em_val = exact_match(hyp, gold)
        r1_val = _rouge_n_score(hyp, gold, 1)
        r2_val = _rouge_n_score(hyp, gold, 2)
        rl_val = rouge_l(hyp, gold)

        all_f1.append(f1_val)
        all_em.append(em_val)
        all_r1.append(r1_val)
        all_r2.append(r2_val)
        all_rl.append(rl_val)

        if qtype not in type_scores:
            type_scores[qtype] = {"f1": [], "em": [], "rouge1": [], "rouge2": [], "rougeL": []}
        type_scores[qtype]["f1"].append(f1_val)
        type_scores[qtype]["em"].append(em_val)
        type_scores[qtype]["rouge1"].append(r1_val)
        type_scores[qtype]["rouge2"].append(r2_val)
        type_scores[qtype]["rougeL"].append(rl_val)

    def _mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    per_type: dict[str, dict[str, float]] = {}
    for qtype, sdict in type_scores.items():
        per_type[qtype] = {metric: round(_mean(vals), 4) for metric, vals in sdict.items()}

    return LongMemEvalScores(
        f1=_mean(all_f1),
        em=_mean(all_em),
        rouge1=_mean(all_r1),
        rouge2=_mean(all_r2),
        rougeL=_mean(all_rl),
        per_type=per_type,
        n_total=len(all_f1),
    )
