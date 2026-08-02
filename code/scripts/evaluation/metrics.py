"""Lexical metrics for the unified evaluation protocol.

Two F1 variants are provided because the literature is split:
- f1_set: set-based token F1, A-Mem utils.py implementation (identical in
  Nemori / SimpleMem). This is the F1 used by all directly comparable papers.
- f1_locomo_official: LoCoMo task_eval/evaluation.py implementation
  (stemmed multiset F1, multi-answer split for cat 1, keyword matching for
  cat 5). Report only as a footnoted extra.

BLEU-1..4 and ROUGE follow A-Mem utils.py (nltk sentence_bleu with
SmoothingFunction().method1; rouge_scorer with stemmer).
"""

import re
import string
from collections import Counter

# --- lazy heavy deps -------------------------------------------------------

_nltk = None
_rouge_scorer = None
_stemmer = None


def _get_nltk():
    global _nltk
    if _nltk is None:
        import nltk
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        _nltk = (nltk, sentence_bleu, SmoothingFunction)
    return _nltk


def _get_rouge():
    global _rouge_scorer
    if _rouge_scorer is None:
        from rouge_score import rouge_scorer
        _rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    return _rouge_scorer


def _get_stemmer():
    global _stemmer
    if _stemmer is None:
        from nltk.stem import PorterStemmer
        _stemmer = PorterStemmer()
    return _stemmer


# =============================================================================
# Set-based F1 (A-Mem utils.py — the comparable-papers implementation)
# =============================================================================

def simple_tokenize(text):
    """A-Mem utils.py verbatim: lower, strip .,!? to spaces, split."""
    text = str(text)
    return (text.lower().replace(".", " ").replace(",", " ")
            .replace("!", " ").replace("?", " ").split())


def f1_set(prediction, reference):
    pred_tokens = set(simple_tokenize(prediction))
    ref_tokens = set(simple_tokenize(reference))
    common = pred_tokens & ref_tokens
    if not pred_tokens or not ref_tokens:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction, reference):
    return int(str(prediction).strip().lower() == str(reference).strip().lower())


# =============================================================================
# LoCoMo official F1 (task_eval/evaluation.py port)
# =============================================================================

def _normalize_answer_official(s):
    s = str(s).replace(",", "")
    s = regex_sub(r"\b(a|an|the|and)\b", " ", remove_punc(s.lower()))
    return " ".join(s.split())


def remove_punc(text):
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)


def regex_sub(pattern, repl, text):
    return re.sub(pattern, repl, text)


def _f1_official_single(prediction, ground_truth):
    ps = _get_stemmer()
    pred_tokens = [ps.stem(w) for w in _normalize_answer_official(prediction).split()]
    gt_tokens = [ps.stem(w) for w in _normalize_answer_official(ground_truth).split()]
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def f1_locomo_official(prediction, reference, category):
    """LoCoMo evaluation.py per-category dispatch.

    cat 1: multi-answer — split both sides on commas, mean over gold parts of
           the best-matching prediction part.
    cat 3: gold uses only the part before ';'.
    cat 5: keyword matching — 1 if prediction contains the abstention phrase.
    """
    prediction, reference = str(prediction), str(reference)
    if category == 5:
        low = prediction.lower()
        return 1.0 if ("no information available" in low or "not mentioned" in low) else 0.0
    if category == 1:
        import numpy as np
        preds = [p.strip() for p in prediction.split(",")]
        golds = [g.strip() for g in reference.split(",")]
        return float(np.mean([max(_f1_official_single(p, g) for p in preds) for g in golds]))
    if category == 3:
        reference = reference.split(";")[0].strip()
    return _f1_official_single(prediction, reference)


# =============================================================================
# BLEU 1-4 (A-Mem utils.py: nltk word_tokenize + method1 smoothing)
# =============================================================================

def bleu_scores(prediction, reference):
    nltk, sentence_bleu, SmoothingFunction = _get_nltk()
    pred_tokens = nltk.word_tokenize(str(prediction).lower())
    ref_tokens = [nltk.word_tokenize(str(reference).lower())]
    weights_list = [(1, 0, 0, 0), (0.5, 0.5, 0, 0),
                    (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]
    smooth = SmoothingFunction().method1
    out = {}
    for n, weights in enumerate(weights_list, start=1):
        try:
            out[f"bleu{n}"] = sentence_bleu(
                ref_tokens, pred_tokens, weights=weights, smoothing_function=smooth)
        except Exception:
            out[f"bleu{n}"] = 0.0
    return out


# =============================================================================
# ROUGE (A-Mem utils.py: rouge_scorer with stemmer)
# =============================================================================

def rouge_scores(prediction, reference):
    scorer = _get_rouge()
    scores = scorer.score(str(reference), str(prediction))
    return {
        "rouge1_f": scores["rouge1"].fmeasure,
        "rouge2_f": scores["rouge2"].fmeasure,
        "rougeL_f": scores["rougeL"].fmeasure,
    }


# =============================================================================
# Per-question metric bundle
# =============================================================================

LEXICAL_METRICS = ("em", "f1", "f1_official", "bleu", "rouge")


def compute_lexical(prediction, reference, category=None,
                    metrics=("em", "f1", "bleu", "rouge")):
    """Compute the requested lexical metrics for one QA pair.

    metrics: subset of LEXICAL_METRICS. 'f1_official' requires category
    (LoCoMo only); ignored gracefully elsewhere.
    """
    out = {}
    if "em" in metrics:
        out["exact_match"] = exact_match(prediction, reference)
    if "f1" in metrics:
        out["f1"] = f1_set(prediction, reference)
    if "f1_official" in metrics and category is not None:
        out["f1_official"] = f1_locomo_official(prediction, reference, category)
    if "bleu" in metrics:
        out.update(bleu_scores(prediction, reference))
    if "rouge" in metrics:
        out.update(rouge_scores(prediction, reference))
    return out
