# AI Recruitment Assistant — Technical Documentation

## 1. Purpose and scope

This repository is an educational prototype for comparing resume text with job requirements. The primary interface uses explainable skill-evidence coverage. An optional neural pipeline demonstrates a Bi-LSTM+CRF entity extractor, a Siamese text-pair model, and a ranking MLP.

**This project is not a validated hiring model.** The score is not a hiring probability, confidence estimate, job-performance prediction, or automated recommendation. Human review is required.

## 2. Main workflow

1. A user selects a predefined role or pastes a custom job description.
2. The user uploads a text-based PDF or pastes resume text.
3. The app recognizes skills using a phrase vocabulary and aliases.
4. It reports matched evidence, missing/unverified skills, score breakdown, and suggestions.
5. Multiple PDFs can be compared with the same role/JD and exported as CSV.
6. An optional experimental tab can run the neural checkpoints separately. Its output is not included in the primary score.

## 3. Primary evidence-coverage score

### Predefined role profile

The score is computed from coverage of the configured skill list:

- 70% weight: required skills found in the resume text.
- 30% weight: preferred skills found in the resume text.

For example, required-skill coverage is the number of recognized required skills found divided by the number of required skills in that profile. Preferred coverage is calculated similarly. The weighted sum is expressed as a percentage.

### Custom job description

The app identifies supported skills mentioned in the JD. Nearby wording cues classify recognized skills as required or preferred. When preferred skills are detected, the score is 70% required-skill coverage + 30% preferred-skill coverage; otherwise it is required-skill coverage. If the JD contains no supported skills, the score is unavailable rather than a fabricated zero.

### Interpretation

- A matched skill means the text contains a recognized phrase or alias.
- A missing/unverified skill means the system did not find recognized evidence in the supplied text. It does not prove the candidate lacks the skill.
- Alias matching and basic negation checks are heuristics. Context, synonyms, skill proficiency, recency, and evidence quality are not fully understood.
- The app does not automatically accept or reject candidates.

## 4. Neural-network components (experimental)

### Bi-LSTM + CRF

A sequence-labeling architecture intended to extract entities such as skills, organizations, education, dates, and experience. Its usefulness depends on the training data and annotation quality.

### Siamese network

A shared-weight text-pair model that produces an experimental similarity signal for a resume and JD. Similarity is not the same as requirement coverage.

### Ranking MLP

A small neural model that combines the Siamese signal with extracted resume features. Its output is displayed only as an experimental model diagnostic and is not calibrated against real hiring outcomes. The pipeline explicitly avoids interpreting the output as confidence or automatically making a hiring decision.

The included training scripts use generated/demo training examples. Successful training or loading weights does not establish generalization, calibration, fairness, or validity. Do not publish performance claims without a documented, held-out, human-labeled evaluation.

## 5. Evaluation

An evaluation template is available at evaluation/labeled_examples_template.csv. Use evaluate_matching.py with a dataset that has been reviewed and labeled by people using a documented rubric.

Metrics are meaningful only relative to the supplied labeled data and evaluation split. Avoid leakage between training and test examples, keep near-duplicate resumes together, and report the dataset source, sample size, class balance, split method, and limitations. Metrics do not establish that the system predicts job performance or is fair.

## 6. Run locally

Use Python 3.10 or 3.11, then:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Run checks:

```powershell
python -m compileall -q .
python -m pytest -q
```

The main rule-based workflow works without neural checkpoints. The experimental neural tab requires ner_model.pth, siamese_model.pth, and ranking_model.pth in the project root. Demo checkpoints can be created with python train_all_models.py.

## 7. Limitations and responsible use

- Only text-based PDFs are supported; scanned/image-only PDFs require OCR, which is not included.
- The skill vocabulary is finite and may miss synonyms or context.
- Custom JDs use nearby wording cues to classify skills as required or preferred; this classification is heuristic and must be reviewed.
- No real-world benchmark, fairness audit, or job-performance validation is included.
- Resume files contain personal information. Process only files you are authorized to use; do not deploy publicly without reviewing access control, data retention, privacy, and security requirements.
- A qualified human should review source documents and job-related evidence before any employment decision.

## 8. Repository map

- app.py — Streamlit UI and CSV exports.
- role_profiles.py — role profiles, skill aliases, and evidence scoring.
- resume_parser.py — PDF text extraction and conservative baseline features.
- inference_pipeline.py and neural model modules — experimental neural pipeline.
- test_evidence_matching.py, test_role_profiles.py, test_evaluate_matching.py — automated tests.
- evaluate_matching.py and evaluation/ — evaluation tooling and labeled-data guidance.
- .github/workflows/tests.yml — CI compile and test checks.
