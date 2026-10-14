from pathlib import Path

import numpy as np

from inference_pipeline import CandidateMatchingPipeline
from model_base import ApplicantScreeningNN, min_max_normalize
from resume_parser import extract_text_from_pdf


ROOT = Path(__file__).resolve().parent


def test_bundled_model_checkpoints_load_and_return_bounded_diagnostics():
    pipeline = CandidateMatchingPipeline(model_dir=str(ROOT))
    assert pipeline.load_models(), "Bundled neural model checkpoints should load"

    result = pipeline.evaluate_candidate(
        "Data analyst with Python, SQL, Excel, data cleaning, data analysis, and Tableau dashboard project.",
        "Data Analyst required skills: Python, SQL, Excel, data cleaning, data analysis, statistics, data visualization.",
    )

    assert 0.0 <= result["siamese_similarity"] <= 1.0
    assert 0.0 <= result["experimental_ranker_output"] <= 1.0
    assert result["confidence"] is None
    assert "REVIEW REQUIRED" in result["decision"]


def test_bundled_sample_pdfs_have_extractable_text():
    for name in ("strong_candidate.pdf", "weak_candidate.pdf"):
        path = ROOT / "sample_resumes" / name
        with path.open("rb") as pdf_file:
            extracted = extract_text_from_pdf(pdf_file)
        assert extracted.strip(), f"Expected readable text in {name}"


def test_baseline_numpy_checkpoint_loads_and_returns_finite_output():
    model_path = ROOT / "model_weights.npz"
    assert model_path.exists(), "Baseline NumPy checkpoint should be present"
    model = ApplicantScreeningNN()
    model.load_model(str(model_path))
    features = np.array([[2.0, 40.0, 30.0, 1.0]], dtype=float)
    normalized, _, _ = min_max_normalize(features, model.min_vals, model.max_vals)
    _, probabilities = model.predict(normalized)
    output = float(probabilities.reshape(-1)[0])
    assert np.isfinite(output)
    assert 0.0 <= output <= 1.0
