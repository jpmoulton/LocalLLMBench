# Verbatim from NVIDIA RULER (Apache-2.0, github.com/NVIDIA/RULER)
# commit c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a
#   scripts/eval/synthetic/constants.py : string_match_part, string_match_all
#   scripts/eval/evaluate.py            : postprocess_pred
# Test fixture only. Not imported by the package; exec'd by tests/test_benchmarks_ruler.py
# so the vendored scorer is checked against upstream's own code rather than a paraphrase.

import re


def postprocess_pred(predict_str: str, task_config: dict):

    predict_str = predict_str.strip()

    # Remove all non-printable characters
    np_pattern = re.compile(r'[\x00-\x1f]')
    predict_str = np_pattern.sub('\n', predict_str).strip()

    return predict_str


def string_match_part(preds, refs):
    score = sum([max([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) for pred, ref in zip(preds, refs)]) / len(preds) * 100
    return round(score, 2)

def string_match_all(preds, refs):
    score = sum([sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref) for pred, ref in zip(preds, refs)]) / len(preds) * 100
    return round(score, 2)
