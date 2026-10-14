# AI Recruitment — Resume Evidence Matching Demo

A Streamlit research/demo project for extracting text from resumes, detecting selected entities, and comparing resume text with a job description. It includes a transparent Data Analyst skill-evidence view and a separate educational neural-network demo.

> **Important limitation:** this project is not a validated hiring system. The bundled neural models were trained with synthetic/template-generated data, and the baseline CSV labels are synthetically generated. Their outputs are experimental and are **not** real-world accuracy, calibrated hiring probabilities, or evidence that a candidate will succeed. Do not automatically shortlist or reject candidates using these scores.

## What it does

- Extracts text from text-based PDF resumes using pypdf.
- Displays a Data Analyst profile with required and preferred skills.
- Checks whether configured skill phrases are present in resume text.
- Reports missing/unverified skill mentions and practical resume-writing suggestions.
- Compares 2–10 PDF resumes against the same target using the same skill-evidence formula.
- Keeps the BiLSTM-CRF, Siamese, deep-ranking, NumPy MLP, and manual-forward-pass demos available for experimentation.
- Includes automated tests for scoring behavior and resume-parser edge cases.

## How the main score works

The main **Skill Evidence Coverage** score is deliberately separate from the experimental neural models.

- **Predefined Data Analyst profile:** 70% required-skill coverage + 30% preferred-skill coverage.
- **Custom job description:** percentage of recognized skill targets named in the description that are also detected in the resume.
- If a custom description contains no recognized skills, the score is shown as **N/A** instead of a misleading 0%.
- A skill counts when an explicit alias/phrase is found in extracted text. This does not prove proficiency, and a missing mention does not prove the candidate lacks the skill.

The neural outputs are retained as diagnostics only. No model-accuracy number is claimed because there is no independent, labeled real-world evaluation set in this repository.

## Setup (Windows PowerShell)

Open a terminal in the repository folder:

\`\`\`powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
\`\`\`

If PowerShell blocks virtual-environment activation, you can run the venv's Python directly without changing execution policy:

\`\`\`powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
\`\`\`

## Run the application

The repository includes the three model checkpoint files and the baseline NumPy weights. From the project root:

\`\`\`powershell
python -m streamlit run app.py
\`\`\`

If a checkpoint is missing or incompatible, the app may fall back to the two educational baseline tabs; the intelligent-matching tab requires all three neural checkpoints.

## Run tests

\`\`\`powershell
python -m pytest -q
\`\`\`

Tests check deterministic score semantics and parser behavior. Passing unit tests verifies code behavior; it does **not** establish model accuracy or hiring validity.

## Optional: retrain the demo neural models

\`\`\`powershell
python train_all_models.py
\`\`\`

This command generates synthetic training examples and overwrites the neural checkpoint files. It can take time on CPU. Retraining is not a way to improve real-world performance without a representative, properly labeled dataset and an independent evaluation split.

## Repository map

- app.py — Streamlit user interface.
- role_profiles.py — role definition, explicit skill aliases, and transparent coverage score.
- resume_parser.py — PDF text extraction and baseline keyword feature extraction.
- inference_pipeline.py — orchestration for the three experimental neural models.
- bilstm_crf_model.py — synthetic-data NER demo.
- siamese_model.py — keyword-vector Siamese matching demo.
- deep_ranking_model.py — ranking network trained on formula-generated synthetic targets.
- model_base.py, train.py, train_candidates.csv, test_candidates.csv — original NumPy MLP demo and synthetic data.
- test_*.py — automated checks.

## Known limitations

- PDF extraction reads embedded text; image-only/scanned resumes need OCR, which is not included.
- Keyword matching can miss synonyms, infer no proficiency, and cannot understand negation or context reliably.
- Resume durations can be ambiguous; extracting multiple date/duration mentions is not a reliable employment-history calculation.
- The current synthetic model training data is suitable for demonstrating a pipeline, not for claiming recruitment performance.
- Before any real-world evaluation, create a consented, representative, human-reviewed dataset; document label definitions; separate train/validation/test data by candidate; measure ranking and classification metrics; inspect errors and subgroup behavior; and calibrate any probability-like output before calling it a probability.
