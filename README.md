# AI Recruitment Assistant

A Streamlit prototype for comparing resume text against role requirements. It combines explainable, rule-based skill evidence coverage with optional experimental neural-network diagnostics.

## What it does

- Assess one candidate from a text-based PDF or pasted resume text.
- Compare multiple PDF resumes against the same predefined role or custom job description.
- Show matched skills, missing/unverified requirements, score breakdowns, and improvement suggestions.
- Export individual assessments and batch comparisons to CSV.
- Offer an optional experimental Bi-LSTM+CRF / Siamese / ranking pipeline for educational demonstration.

## Setup (Windows PowerShell)

Use Python 3.10 or 3.11 for the broadest dependency compatibility.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If the py launcher is unavailable, create the environment with python -m venv .venv, activate it, and continue with the remaining commands.

## Run tests

```powershell
python -m pytest -q
python -m compileall -q .
```

The optional neural tab needs the three trained checkpoints in the project root: ner_model.pth, siamese_model.pth, and ranking_model.pth. If they are not present, the primary rule-based assessment still works. To create demo checkpoints, run:

```powershell
python train_all_models.py
```

The training data and generated checkpoints are educational/demo artifacts, not a validated recruitment model.

## Score semantics

**Predefined role:** 70% required-skill coverage + 30% preferred-skill coverage.

**Custom job description:** recognized skills are classified as required or preferred using nearby wording cues. When preferred skills are detected, the score is 70% required-skill coverage + 30% preferred-skill coverage; otherwise it is required-skill coverage. If no supported skills are recognized, the score is N/A.

Scores measure recognized textual evidence coverage only. They are not probabilities, confidence values, hiring recommendations, or estimates of performance. A missing skill means the parser did not find evidence in the provided text; it does not establish that the person lacks that skill.

## Current limitations

- Skill recognition uses phrase aliases and simple negation checks; it is not full semantic understanding.
- Only text-based PDFs are supported. OCR for scanned/image-only PDFs is not included.
- Custom job descriptions use local wording cues to heuristically distinguish required from preferred skills; reviewers should verify this classification.
- Experimental neural checkpoints are not calibrated on real-world recruitment outcomes.
- No claims of fairness, predictive validity, or suitability for real hiring decisions are made.
- Use only resumes you are authorized to process. A public deployment needs a privacy, retention, access-control, and security review.

## Repository map

- app.py — Streamlit UI, single-candidate and batch assessment.
- role_profiles.py — role definitions, aliases, and explainable scoring.
- resume_parser.py — PDF text extraction and conservative baseline features.
- inference_pipeline.py and neural model modules — experimental model path.
- test_evidence_matching.py, test_role_profiles.py — matching tests.
- evaluation/ — instructions and tools for evaluating on human-labeled data.

## Responsible use

This is an educational prototype. Always have a qualified human review the underlying resume and job criteria. Do not use this application to automatically reject applicants or as the sole basis for employment decisions.
