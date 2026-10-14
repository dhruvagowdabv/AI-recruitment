"""Evaluate skill-evidence ranking against human-labeled resume/JD pairs.

Expected CSV columns: resume_text, job_description, label (0 or 1).
No bundled demo labels are treated as ground truth.
"""
import argparse
import csv
import math
from collections import defaultdict

from role_profiles import analyze_role_match


def _auc(labels, scores):
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ordered = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum = 0.0
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        average_rank = ((i + 1) + j) / 2.0
        rank_sum += average_rank * sum(label for _, label in ordered[i:j])
        i = j
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _average_precision(labels, scores):
    positives = sum(labels)
    if positives == 0:
        return None
    ranked = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    found = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(ranked, 1):
        if label:
            found += 1
            precision_sum += found / rank
    return precision_sum / positives


def _precision_recall_at_k(labels, scores, k):
    ranked = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)[:k]
    relevant_found = sum(label for _, label in ranked)
    positives = sum(labels)
    precision = relevant_found / len(ranked) if ranked else 0.0
    recall = relevant_found / positives if positives else None
    return precision, recall


def evaluate_rows(rows, k=5):
    groups = defaultdict(list)
    for row in rows:
        resume = row.get("resume_text", "")
        jd = row.get("job_description", "")
        label_raw = str(row.get("label", "")).strip()
        if not resume.strip() or not jd.strip():
            raise ValueError("Each row must include non-empty resume_text and job_description.")
        if label_raw not in {"0", "1"}:
            raise ValueError("Each label must be 0 or 1; received {!r}.".format(label_raw))
        result = analyze_role_match(resume, jd, use_profile=False)
        if not result["score_available"]:
            raise ValueError(
                "Row has no recognized target skills in its job description. "
                "Extend the vocabulary or remove that row before evaluating."
            )
        groups[jd].append((int(label_raw), float(result["role_match_percentage"])))

    report = {"examples": sum(len(group) for group in groups.values()), "jobs": []}
    auc_values, ap_values = [], []
    for jd, pairs in groups.items():
        labels = [p[0] for p in pairs]
        scores = [p[1] for p in pairs]
        precision, recall = _precision_recall_at_k(labels, scores, k)
        report["jobs"].append({
            "job_description_preview": jd[:100].replace("\n", " "),
            "examples": len(pairs),
            "positive_labels": sum(labels),
            "roc_auc": _auc(labels, scores),
            "average_precision": _average_precision(labels, scores),
            "precision_at_k": precision,
            "recall_at_k": recall,
            "k": min(k, len(pairs)),
        })
        if report["jobs"][-1]["roc_auc"] is not None:
            auc_values.append(report["jobs"][-1]["roc_auc"])
        if report["jobs"][-1]["average_precision"] is not None:
            ap_values.append(report["jobs"][-1]["average_precision"])
    report["overall"] = {
        "macro_roc_auc_by_job": sum(auc_values) / len(auc_values) if auc_values else None,
        "macro_average_precision_by_job": sum(ap_values) / len(ap_values) if ap_values else None,
        "note": "Macro averages are across job-description groups with defined metrics. Results describe only this labeled file and do not establish hiring validity or fairness.",
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="CSV containing resume_text, job_description, label")
    parser.add_argument("--k", type=int, default=5, help="K for precision@K and recall@K (default: 5)")
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be at least 1")
    with open(args.csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"resume_text", "job_description", "label"}
    if not rows:
        parser.error("CSV has no data rows.")
    if not required.issubset(rows[0]):
        parser.error("CSV must contain columns: resume_text, job_description, label")
    import json
    print(json.dumps(evaluate_rows(rows, k=args.k), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
