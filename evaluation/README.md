# Evaluation with labeled examples

Do not report model accuracy until the system has been evaluated against a held-out dataset with human-reviewed labels.

Create a CSV with these columns:

- resume_text: extracted resume text.
- job_description: target job description.
- label: human-reviewed relevance label, 1 for relevant and 0 for not relevant.

Use examples collected with appropriate authorization and a documented labeling rubric. Keep candidates from the same person and near-duplicate resumes in only one split. Labelers should judge role-related evidence using the job criteria, not demographic or protected characteristics.

Run:

```bash
python evaluate_matching.py path/to/labeled_examples.csv --k 5
```

The script reports ROC AUC (only when both classes exist), average precision, precision@K and recall@K per job description, plus the number of labeled examples. These are measurements on the supplied dataset only; they do not establish job-performance validity, fairness, or generalization to other roles. Use a held-out test split and document its provenance before sharing metrics.
