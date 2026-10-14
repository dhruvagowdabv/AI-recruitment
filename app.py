"""Streamlit UI for explainable, human-reviewed resume-to-role comparison."""
import io
import os

import pandas as pd
import streamlit as st

from resume_parser import extract_text_from_pdf, parse_resume
from role_profiles import (
    ROLE_PROFILES,
    analyze_role_match,
    build_role_description,
    get_role_profile,
)

st.set_page_config(
    page_title="AI Recruitment Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _read_pdf(uploaded_file):
    """Extract PDF text and return (text, error)."""
    try:
        text = extract_text_from_pdf(uploaded_file)
    except Exception as exc:
        return "", "Could not read this PDF: {}".format(exc)
    if not text.strip():
        return "", "No selectable text was found. This may be a scanned PDF; OCR is not currently supported."
    return text, None


def _assessment_row(name, resume_text, target_text, use_profile, role_name):
    analysis = analyze_role_match(
        resume_text=resume_text,
        target_text=target_text,
        use_profile=use_profile,
        role_name=role_name,
    )
    return {
        "Candidate": name,
        "Evidence coverage (%)": analysis["role_match_percentage"],
        "Matched skills": ", ".join(analysis["matched_skills"]),
        "Missing required skills": ", ".join(analysis["missing_required_skills"]),
        "Missing preferred skills": ", ".join(analysis["missing_preferred_skills"]),
        "analysis": analysis,
    }


def main():
    st.title("AI Recruitment Assistant")
    st.write(
        "Compare resume evidence against job requirements. Scores represent textual skill coverage, "
        "not a hiring probability or a prediction of future job performance."
    )
    st.info(
        "Human review required. A skill not found in the resume text is treated as unverified, "
        "not proof that the candidate lacks it. Do not use this tool as the sole basis for hiring decisions."
    )

    with st.sidebar:
        st.header("Role setup")
        role_name = st.selectbox("Predefined role", list(ROLE_PROFILES.keys()))
        profile = get_role_profile(role_name)
        st.caption(profile["description"])
        custom_jd = st.text_area(
            "Optional custom job description",
            height=210,
            placeholder="Paste the job description. Recognized skills in this text become the comparison targets.",
        )
        use_profile = not bool(custom_jd.strip())
        target_text = build_role_description(profile) if use_profile else custom_jd.strip()
        if use_profile:
            st.caption("Using the predefined role profile.")
        else:
            st.caption("Using recognized skills from your custom job description.")

    single_tab, batch_tab, model_tab, about_tab = st.tabs(
        ["Single candidate", "Batch comparison", "Experimental neural models", "Method & limitations"]
    )

    with single_tab:
        left, right = st.columns([1, 1])
        with left:
            st.subheader("Resume")
            input_method = st.radio("Input method", ["Upload PDF", "Paste text"], horizontal=True)
            resume_text = ""
            candidate_name = "Candidate"
            if input_method == "Upload PDF":
                uploaded = st.file_uploader("Upload a text-based PDF", type=["pdf"], key="single_pdf")
                if uploaded is not None:
                    candidate_name = os.path.splitext(uploaded.name)[0]
                    resume_text, error = _read_pdf(uploaded)
                    if error:
                        st.error(error)
                    else:
                        st.success("Extracted {} words.".format(len(resume_text.split())))
            else:
                candidate_name = st.text_input("Candidate label", value="Candidate")
                resume_text = st.text_area(
                    "Paste resume text",
                    height=280,
                    placeholder="Paste the resume's text here...",
                )

        with right:
            st.subheader("Assessment")
            st.write("**Target:** {}".format(role_name if use_profile else "Custom job description"))
            st.write("**Required skills:** " + (", ".join(profile["required_skills"]) if use_profile else "Recognized skills from the custom JD"))
            st.write("**Preferred skills:** " + (", ".join(profile["preferred_skills"]) if use_profile else "Detected from preferred/bonus wording in the JD; verify manually"))
            assess = st.button("Assess resume", type="primary", use_container_width=True)

            if assess:
                if not resume_text.strip():
                    st.warning("Upload a readable PDF or paste resume text first.")
                else:
                    result = analyze_role_match(
                        resume_text=resume_text,
                        target_text=target_text,
                        use_profile=use_profile,
                        role_name=role_name,
                    )
                    if result["score_available"]:
                        st.metric("Evidence coverage", "{}%".format(result["role_match_percentage"]))
                        st.progress(result["role_match_percentage"] / 100.0)
                    else:
                        st.metric("Evidence coverage", "N/A")
                        st.warning("This job description contains no recognized target skills, so a score would be misleading.")
                    st.caption(result["score_note"])
                    c1, c2 = st.columns(2)
                    c1.metric("Matched skills", "{}/{}".format(result["matched_skill_count"], result["target_skill_count"]))
                    c2.metric("Required gaps", str(len(result["missing_required_skills"])))
                    st.markdown("#### Skill evidence")
                    st.write("**Matched:** " + (", ".join(result["matched_skills"]) or "No recognized skills found"))
                    st.write("**Missing/unverified required:** " + (", ".join(result["missing_required_skills"]) or "None detected"))
                    st.write("**Missing/unverified preferred:** " + (", ".join(result["missing_preferred_skills"]) or "None detected"))
                    st.markdown("#### Score breakdown")
                    st.dataframe(pd.DataFrame(result["score_breakdown"]), hide_index=True, use_container_width=True)
                    st.markdown("#### Suggestions")
                    for suggestion in result["improvement_suggestions"]:
                        st.markdown("- " + suggestion)
                    st.download_button(
                        "Download assessment (CSV)",
                        data=pd.DataFrame([{
                            "Candidate": candidate_name,
                            "Role": role_name,
                            "Scoring mode": result["scoring_mode"],
                            "Evidence coverage (%)": result["role_match_percentage"],
                            "Matched skills": ", ".join(result["matched_skills"]),
                            "Missing required skills": ", ".join(result["missing_required_skills"]),
                            "Missing preferred skills": ", ".join(result["missing_preferred_skills"]),
                            "Score note": result["score_note"],
                        }]).to_csv(index=False).encode("utf-8"),
                        file_name="candidate_assessment.csv",
                        mime="text/csv",
                    )

    with batch_tab:
        st.subheader("Compare multiple candidates")
        st.write("Upload several PDFs and rank them by the same evidence-coverage method.")
        uploads = st.file_uploader(
            "Upload candidate PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            key="batch_pdfs",
        )
        if st.button("Compare candidates", type="primary", key="compare_batch"):
            if len(uploads) < 2:
                st.warning("Upload at least two readable PDF resumes.")
            else:
                rows = []
                errors = []
                for upload in uploads:
                    text, error = _read_pdf(upload)
                    if error:
                        errors.append("{}: {}".format(upload.name, error))
                        continue
                    rows.append(_assessment_row(
                        os.path.splitext(upload.name)[0], text, target_text, use_profile, role_name
                    ))
                for error in errors:
                    st.warning(error)
                if len(rows) < 2:
                    st.error("At least two resumes must be readable to compare candidates.")
                else:
                    table = pd.DataFrame([{
                        "Candidate": row["Candidate"],
                        "Evidence coverage (%)": row["Evidence coverage (%)"],
                        "Matched skills": row["Matched skills"],
                        "Missing required skills": row["Missing required skills"],
                        "Missing preferred skills": row["Missing preferred skills"],
                    } for row in rows])
                    if table["Evidence coverage (%)"].notna().any():
                        table = table.sort_values(
                            "Evidence coverage (%)", ascending=False, na_position="last"
                        ).reset_index(drop=True)
                        table.insert(0, "Order by evidence coverage", range(1, len(table) + 1))
                    st.dataframe(table, hide_index=True, use_container_width=True)
                    st.caption("Ordering is a convenience for human review, not an automated hiring recommendation.")
                    st.download_button(
                        "Download comparison (CSV)",
                        data=table.to_csv(index=False).encode("utf-8"),
                        file_name="candidate_comparison.csv",
                        mime="text/csv",
                    )

    with model_tab:
        st.subheader("Experimental deep-learning pipeline")
        st.warning(
            "These neural models are experimental and trained with synthetic/demo data. Their outputs are not calibrated, "
            "are not included in the primary evidence-coverage score, and must not be interpreted as hiring probabilities."
        )
        neural_resume = st.text_area("Resume text for experimental models", height=180, key="neural_resume")
        if st.button("Run experimental pipeline", key="run_neural"):
            if not neural_resume.strip():
                st.warning("Paste resume text first.")
            elif not target_text.strip():
                st.warning("Provide a job description first.")
            else:
                try:
                    from inference_pipeline import CandidateMatchingPipeline
                    pipeline = CandidateMatchingPipeline()
                    with st.spinner("Loading model checkpoints..."):
                        loaded = pipeline.load_models()
                    if not loaded:
                        st.error("One or more model checkpoints are missing. See setup instructions in README.md.")
                    else:
                        result = pipeline.evaluate_candidate(neural_resume, target_text)
                        st.metric("Experimental ranking output", "{}%".format(result["match_percentage"]))
                        st.caption("Uncalibrated model output for demonstration only; do not use as a hiring decision.")
                        st.write("**NER entities:**")
                        st.json(result["entities"])
                        st.write("**Siamese similarity signal:** {}".format(result["siamese_similarity"]))
                        st.write("**Model feature vector:**")
                        st.json(result["feature_vector"])
                except Exception as exc:
                    st.error("Experimental model execution failed: {}".format(exc))

    with about_tab:
        st.subheader("How the score works")
        st.markdown(
            """
            - **Predefined profile:** 70% of the score comes from required-skill coverage and 30% from preferred-skill coverage.
            - **Custom job description:** score is the fraction of recognized skills in the job description that are also found in the resume.
            - **No recognized target skills:** score is unavailable instead of showing a fabricated zero.
            - **Evidence matching:** phrase aliases and basic negation checks are used. This is still a keyword-based heuristic, not deep semantic understanding.
            - **No automatic rejection:** the application reports evidence and gaps; a person must review context and make decisions.
            - **PDF limitations:** text-based PDFs are supported. Scanned/image-only PDFs need OCR, which is not included.
            - **Privacy:** process only resumes you are authorized to use. Avoid uploading sensitive personal data to public deployments.
            """
        )
        st.subheader("Current limitations")
        st.markdown(
            """
            - Skill recognition can miss synonyms, nuanced context, negation beyond simple patterns, and skills shown only in images.
            - Custom JDs do not automatically classify each requirement as required or preferred.
            - The included neural models are experimental; their training data is synthetic/demo data and no real-world recruitment benchmark is bundled.
            - No fairness or job-performance validity claim is made. Evaluate against appropriately labeled, consented data before any real-world use.
            """
        )


if __name__ == "__main__":
    main()
