"""
AI Recruitment Simulation System - Enhanced Streamlit Dashboard
================================================================
Integrates all 3 custom neural networks:
    Tab 1: Intelligent Candidate Matching (BiLSTM-CRF + Siamese + Deep Ranking)
    Tab 2: Resume Screening (Original MLP)
    Tab 3: Manual Simulation (Original sliders + batch leaderboard)
"""

import streamlit as st
import numpy as np
import pandas as pd
import os
import sys
import json

# Original model imports
from model_base import ApplicantScreeningNN, min_max_normalize
from resume_parser import parse_resume
from role_profiles import (
    get_role_profile,
    build_role_description,
    analyze_role_match,
)

st.set_page_config(
    page_title="AI Recruitment Simulator - Neural Matching", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Model Loading
# ============================================================
@st.cache_resource
def load_original_model():
    """Load the original NumPy MLP model."""
    model_path = 'model_weights.npz'
    if not os.path.exists(model_path):
        return None
    nn = ApplicantScreeningNN()
    nn.load_model(model_path)
    return nn

@st.cache_resource
def load_neural_pipeline():
    """Load all 3 custom neural network models."""
    try:
        from inference_pipeline import CandidateMatchingPipeline
        pipeline = CandidateMatchingPipeline()
        if pipeline.load_models():
            return pipeline
        return None
    except Exception as e:
        st.warning(f"Could not load neural matching pipeline: {e}")
        return None


def manual_forward_pass(nn, X_raw):
    """Executes a manual forward pass for demonstration and returns metrics."""
    # Ensure 2D array
    X_raw = np.array(X_raw).reshape(1, -1)
    
    # 1. Normalize
    X_norm, _, _ = min_max_normalize(X_raw, nn.min_vals, nn.max_vals)
    
    # 2. Forward Propagation steps
    Z1 = np.dot(X_norm, nn.W1) + nn.b1
    A1 = nn.relu(Z1)
    
    Z2 = np.dot(A1, nn.W2) + nn.b2
    A2 = nn.sigmoid(Z2)
    
    score = float(A2[0][0]) * 100
    decision = "Review manually — demonstration only"
    
    return {
        "X_norm": X_norm.flatten(),
        "Z1": Z1.flatten(),
        "A1": A1.flatten(),
        "Z2": Z2.flatten(),
        "Fit Score": score,
        "Decision": decision
    }


# ============================================================
# Main Application
# ============================================================
def main():
    # Sidebar
    st.sidebar.title("AI Recruitment System")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Custom Neural Networks")
    st.sidebar.markdown("""
    **1. Bi-LSTM + CRF**  
    Named Entity Recognition
    
    **2. Siamese Network**  
    Experimental keyword-vector pair signal
    
    **3. Deep Ranking MLP**  
    Raw output trained on synthetic targets
    """)
    st.sidebar.markdown("---")
    st.sidebar.markdown("Built with PyTorch + NumPy from scratch")
    
    # Title
    st.title("AI Recruitment Simulation System")
    st.caption("Resume evidence analysis and experimental neural-network demonstrations")
    st.warning(
        "Research/demo project: model outputs are not validated hiring probabilities or model accuracy. "
        "Use results to support human review only; do not automatically reject or shortlist candidates."
    )
    
    # Load models
    nn = load_original_model()
    pipeline = load_neural_pipeline()
    
    # Create tabs
    if pipeline:
        tab1, tab2, tab3 = st.tabs([
            "Intelligent Matching (3 Networks)", 
            "Resume Screening (PDF)", 
            "Manual Simulation"
        ])
    else:
        tab2, tab3 = st.tabs([
            "Resume Screening (PDF)", 
            "Manual Simulation"
        ])
        tab1 = None
        st.info("Neural matching models not found. Run `python train_all_models.py` to enable Tab 1.")
    
    # ================================================================
    # TAB 1: Intelligent Candidate Matching (3 Neural Networks)
    # ================================================================
    if tab1 and pipeline:
        with tab1:
            st.header("Intelligent Candidate Matching")
            st.markdown("The three-network pipeline is retained for experimentation. The primary score below is transparent skill-evidence coverage; neural outputs are uncalibrated diagnostics.")
            
            # Two columns: JD input + Resume input
            col_jd, col_resume = st.columns(2)
            
            with col_jd:
                st.subheader("Target Role")
                selected_role = st.selectbox(
                    "Choose a role",
                    ["Data Analyst"],
                    key="target_role",
                    help="More predefined roles can be added without changing the matching workflow.",
                )
                role_profile = get_role_profile(selected_role)
                st.caption("A predefined role profile is used when no custom job description is supplied.")
                with st.expander("View skills included in the Data Analyst profile", expanded=True):
                    st.markdown("**Core / required skills**")
                    st.write(", ".join(role_profile["required_skills"]))
                    st.markdown("**Preferred / additional skills**")
                    st.write(", ".join(role_profile["preferred_skills"]))
                    st.caption(
                        "To tailor the assessment, paste the company's job description below. "
                        "Add role-specific tools, domain knowledge, or responsibilities there; "
                        "recognized skills mentioned in the custom description are used for skill matching."
                    )
                jd_text = st.text_area(
                    "Optional: paste a specific job description",
                    height=200,
                    placeholder="Paste a company's Data Analyst job description to tailor the assessment. Leave blank to use the predefined role profile.",
                    key="jd_input"
                )
                effective_jd = jd_text.strip() or build_role_description(role_profile)
                if not jd_text.strip():
                    st.info("Using the predefined Data Analyst role profile.")
            
            with col_resume:
                st.subheader("Resume Input")
                input_mode = st.radio("Input method:", ["Upload PDF", "Paste Text"], horizontal=True, key="input_mode")
                
                resume_text = ""
                if input_mode == "Upload PDF":
                    uploaded_file = st.file_uploader("Upload Resume PDF", type="pdf", key="pdf_neural")
                    if uploaded_file is not None:
                        try:
                            from resume_parser import extract_text_from_pdf
                            resume_text = extract_text_from_pdf(uploaded_file)
                            st.success(f"PDF parsed ({len(resume_text.split())} words extracted)")
                        except Exception as e:
                            st.error(f"Error parsing PDF: {e}")
                else:
                    resume_text = st.text_area(
                        "Paste resume text",
                        height=150,
                        placeholder="e.g., Experienced data scientist with 5 years in Python, TensorFlow, machine learning. Worked at Google. Master's in Computer Science from Stanford.",
                        key="resume_text_input"
                    )
            
            st.markdown("---")
            
            # Evaluate button
            if st.button("Evaluate Candidate", type="primary", use_container_width=True, key="eval_btn"):
                if not resume_text.strip():
                    st.warning("Please provide a resume (upload PDF or paste text).")
                else:
                    with st.spinner("Running the neural matching pipeline and role-specific analysis..."):
                        result = pipeline.evaluate_candidate(resume_text, effective_jd)
                        role_analysis = analyze_role_match(
                            resume_text=resume_text,
                            target_text=effective_jd,
                            model_result=result,
                            use_profile=not bool(jd_text.strip()),
                            role_name=selected_role,
                        )
                    
                    # ---- Results Display ----
                    st.markdown("---")
                    st.subheader(f"{selected_role} Assessment")
                    st.caption(
                        "This score measures configured skill evidence found in the resume text. "
                        "It is not model accuracy, a probability of job success, or proof of proficiency."
                    )
                    score_col, coverage_col, target_col = st.columns(3)
                    score_col.metric(
                        "Skill Evidence Coverage",
                        f"{role_analysis['role_match_percentage']:.1f}/100"
                        if role_analysis["score_available"] else "N/A",
                    )
                    if not role_analysis["score_available"]:
                        st.warning(
                            "No recognized skills were found in the custom job description, so a skill-coverage score cannot be calculated. "
                            "Add explicit tools/skills to the JD or use the predefined role profile."
                        )
                    coverage_col.metric(
                        "Target Skills Found",
                        f"{role_analysis['matched_skill_count']}/{role_analysis['target_skill_count']}",
                    )
                    target_col.metric(
                        "Experimental Pair Signal",
                        f"{result['siamese_similarity']:.3f}",
                    )
                    st.info(
                        "Use this as a review aid, not an automatic shortlist/reject decision. "
                        "A missing skill means it was not detected in the text; it does not prove the candidate lacks it."
                    )
                    
                    breakdown_col, skills_col = st.columns([1, 1])
                    with breakdown_col:
                        st.markdown("#### Score Breakdown")
                        st.dataframe(
                            pd.DataFrame(role_analysis["score_breakdown"]),
                            hide_index=True,
                            use_container_width=True,
                        )
                    with skills_col:
                        st.markdown("#### Matched Skills")
                        if role_analysis["matched_skills"]:
                            st.write(", ".join(role_analysis["matched_skills"]))
                        else:
                            st.write("No target skills were detected in the resume.")
                        st.markdown("#### Missing or Unverified Skills")
                        if role_analysis["missing_skills"]:
                            st.write(", ".join(role_analysis["missing_skills"]))
                        else:
                            st.write("All skills in this target profile were detected.")
                    
                    st.markdown("#### Improvement Suggestions")
                    for suggestion in role_analysis["improvement_suggestions"]:
                        st.markdown(f"- {suggestion}")
                    
                    with st.expander("How the role score is calculated"):
                        st.write(
                            "For the predefined Data Analyst profile, evidence coverage is calculated as "
                            "70% required-skill coverage plus 30% preferred-skill coverage. For a custom "
                            "job description, it is the proportion of recognized skills named in that "
                            "description that appear in the resume text. This is a text-matching heuristic, "
                            "not model accuracy, a hiring probability, or proof that a skill is proficiently used."
                        )
                    
                    st.markdown("---")
                    st.subheader("Existing Neural Network Outputs")
                    st.caption(
                        "Experimental diagnostics from models trained on synthetic/demo data. "
                        "These values are shown for transparency only and are not validated match probabilities."
                    )
                    st.metric("Experimental ranker output (0–1)", f"{result['match_score']:.3f}")
                    
                    # Three columns for the 3 network outputs
                    st.markdown("### Neural Network Outputs")
                    net1, net2, net3 = st.columns(3)
                    
                    with net1:
                        st.markdown("#### 1. Bi-LSTM + CRF (NER)")
                        st.caption("Extracted Entities from Resume")
                        entities = result["entities"]
                        
                        if entities.get("skills"):
                            st.markdown(f"**Skills:** {', '.join(entities['skills'])}")
                        else:
                            st.markdown("**Skills:** _None detected_")
                        
                        if entities.get("organizations"):
                            st.markdown(f"**Organizations:** {', '.join(entities['organizations'])}")
                        else:
                            st.markdown("**Organizations:** _None detected_")
                        
                        if entities.get("education"):
                            st.markdown(f"**Education:** {', '.join(entities['education'])}")
                        else:
                            st.markdown("**Education:** _None detected_")
                        
                        if entities.get("experience"):
                            st.markdown(f"**Experience:** {', '.join(entities['experience'])}")
                        else:
                            st.markdown("**Experience:** _None detected_")
                        
                        if entities.get("dates"):
                            st.markdown(f"**Dates:** {', '.join(entities['dates'])}")
                        else:
                            st.markdown("**Dates:** _None detected_")
                    
                    with net2:
                        st.markdown("#### 2. Siamese Network")
                        st.caption("Experimental similarity signal (not calibrated)")
                        
                        sim_score = result["siamese_similarity"]
                        st.metric("Semantic Similarity", f"{sim_score:.4f}")
                        st.progress(min(sim_score, 1.0))
                        
                        st.caption(
                            "The Siamese model uses keyword-frequency vectors and synthetic training pairs; "
                            "treat this as an experimental signal, not a validated semantic similarity score."
                        )
                    
                    with net3:
                        st.markdown("#### 3. Deep Ranking MLP")
                        st.caption("Experimental neural output (not calibrated)")
                        
                        fv = result["feature_vector"]
                        st.metric("Raw ranker output (0–1)", f"{result['match_score']:.3f}")
                        st.progress(max(0.0, min(float(result['match_score']), 1.0)))
                        
                        st.markdown("**Input Feature Vector:**")
                        fv_df = pd.DataFrame({
                            "Feature": ["Siamese Similarity", "Skill Count", "Experience (Yrs)", 
                                       "Education Level", "Org Count", "Project Count"],
                            "Value": [fv["siamese_similarity"], fv["skill_count"], fv["experience_years"],
                                     fv["education_level"], fv["org_count"], fv["project_count"]]
                        })
                        st.dataframe(fv_df, hide_index=True, use_container_width=True)
            
            # ---- Batch Comparison ----
            st.markdown("---")
            st.subheader("Batch Candidate Comparison")
            st.caption("Upload 2–10 PDF resumes. Candidates are ranked by transparent skill-evidence coverage, not the experimental neural score.")

            batch_files = st.file_uploader(
                "Upload candidate resumes (PDF)",
                type=["pdf"],
                accept_multiple_files=True,
                key="batch_resume_pdfs",
                help="Select multiple PDF files at once. The filename is used as the candidate name.",
            )
            if batch_files and len(batch_files) > 10:
                st.warning("Please upload no more than 10 resumes per comparison.")

            if batch_files:
                st.write("**Files selected:**")
                for file in batch_files[:10]:
                    st.write(f"- {file.name}")

            if st.button("Rank All Candidates", key="rank_btn"):
                if not batch_files or len(batch_files) < 2:
                    st.warning("Please upload at least 2 PDF resumes to create a leaderboard.")
                elif len(batch_files) > 10:
                    st.warning("Please upload no more than 10 PDF resumes per comparison.")
                else:
                    from resume_parser import extract_text_from_pdf
                    candidates = []
                    parse_errors = []
                    for file in batch_files:
                        try:
                            text = extract_text_from_pdf(file)
                            if not text.strip():
                                parse_errors.append(f"{file.name}: no readable text found. It may be a scanned PDF.")
                                continue
                            candidates.append({
                                "name": os.path.splitext(file.name)[0],
                                "resume_text": text,
                            })
                        except Exception as exc:
                            parse_errors.append(f"{file.name}: could not read PDF ({exc})")

                    for error in parse_errors:
                        st.warning(error)

                    if len(candidates) < 2:
                        st.error("At least 2 readable PDF resumes are required to rank candidates.")
                    else:
                        with st.spinner("Extracting PDF text and comparing skill evidence..."):
                            ranked_results = []
                            for candidate in candidates:
                                result = pipeline.evaluate_candidate(candidate["resume_text"], effective_jd)
                                analysis = analyze_role_match(
                                    resume_text=candidate["resume_text"],
                                    target_text=effective_jd,
                                    model_result=result,
                                    use_profile=not bool(jd_text.strip()),
                                    role_name=selected_role,
                                )
                                ranked_results.append({
                                    "name": candidate["name"],
                                    "analysis": analysis,
                                    "diagnostics": result,
                                })
                            ranked_results.sort(
                                key=lambda item: (
                                    item["analysis"]["role_match_percentage"]
                                    if item["analysis"]["role_match_percentage"] is not None
                                    else -1.0
                                ),
                                reverse=True,
                            )

                        st.markdown("### Ranked Leaderboard")
                        st.caption("Rank is based on the same evidence-coverage formula shown in the single-candidate assessment. N/A means the custom JD contained no recognized skills.")
                        leaderboard_data = []
                        for rank, item in enumerate(ranked_results, start=1):
                            analysis = item["analysis"]
                            leaderboard_data.append({
                                "Rank": rank,
                                "Candidate": item["name"],
                                "Evidence Coverage (%)": (
                                    analysis["role_match_percentage"]
                                    if analysis["score_available"] else "N/A"
                                ),
                                "Target Skills Found": f"{analysis['matched_skill_count']}/{analysis['target_skill_count']}",
                                "Missing / Unverified Skills": ", ".join(analysis["missing_skills"]) or "None detected",
                                "Experimental Ranker (0–1)": round(item["diagnostics"]["match_score"], 3),
                            })

                        df_lb = pd.DataFrame(leaderboard_data)
                        st.dataframe(df_lb, hide_index=True, use_container_width=True)
                        st.info(
                            "Leaderboard order reflects detected skill mentions only. Review each resume manually; "
                            "PDF text extraction and keyword matching can miss skills or misinterpret context."
                        )
    
    # ================================================================
    # TAB 2: Original Resume Screening
    # ================================================================
    with tab2:
        st.header("Resume Feature Extraction (Demo)")
        st.caption("Keyword counts and the baseline network are educational approximations, not validated candidate assessments. Duration mentions may overlap and are not a verified employment-history calculation.")
        
        if nn is None:
            st.error("Model file model_weights.npz not found! Please run train.py first.")
            return
        
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="pdf_original")
        
        if uploaded_file is not None:
            try:
                with st.spinner("Parsing resume..."):
                    features = parse_resume(uploaded_file)
            except Exception as exc:
                st.error(f"Could not parse this PDF: {exc}")
                st.stop()

            if not features.get("Text_Extracted", True):
                st.error("No readable text was extracted from this PDF. It may be scanned or image-only; use a text-based PDF.")
                st.stop()

            st.subheader("Extracted Candidate Features")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Duration Mentions (Estimate)", f"{features['Experience']} Yrs")
            col2.metric("Technical Score", f"{features['Tech_Score']}/100")
            col3.metric("Soft Skills Score", f"{features['Soft_Score']}/100")
            col4.metric("Projects Completed", f"{features['Projects']}")
            
            # Forward Pass
            X_raw = [features['Experience'], features['Tech_Score'], features['Soft_Score'], features['Projects']]
            results = manual_forward_pass(nn, X_raw)
            
            st.divider()
            st.subheader("Neural Network Inference (Forward Pass)")
            
            with st.expander("Show Matrix Math (Under the hood)"):
                st.write("**Normalized Input (X):**", np.round(results["X_norm"], 4))
                st.write("**Hidden Layer Pre-activation (Z1 = X.W1 + b1):**", np.round(results["Z1"], 4))
                st.write("**Hidden Layer Activation (A1 = ReLU(Z1)):**", np.round(results["A1"], 4))
                st.write("**Output Layer Pre-activation (Z2 = A1.W2 + b2):**", np.round(results["Z2"], 4))
                
            st.divider()
            st.subheader("Demonstration Output")
            st.metric("Uncalibrated model output (0–100 scale)", f"{results['Fit Score']:.2f}")
            st.info("No shortlist/reject decision is made. This score is not calibrated and must not be interpreted as accuracy or probability.")
                
    # ================================================================
    # TAB 3: Manual Simulation
    # ================================================================
    with tab3:
        st.header("Manual Model Simulation")
        st.caption("Change the inputs to inspect the baseline neural network. Its output is a demonstration score, not a hiring probability.")
        
        if nn is None:
            st.error("Model file model_weights.npz not found!")
            return
        
        col_slider1, col_slider2 = st.columns(2)
        with col_slider1:
            exp_val = st.slider("Years of Experience", 0.0, 10.0, 3.0, 0.5)
            tech_val = st.slider("Technical Score", 0, 100, 60, 1)
        with col_slider2:
            soft_val = st.slider("Soft Skills Score", 0, 100, 50, 1)
            proj_val = st.slider("Projects Completed", 0, 6, 2, 1)
            
        X_raw = [exp_val, tech_val, soft_val, proj_val]
        results = manual_forward_pass(nn, X_raw)
        
        st.markdown("### Demonstration Output")
        st.metric("Uncalibrated model output (0–100 scale)", f"{results['Fit Score']:.2f}")
        st.info("This is a model output for experimentation, not a validated hiring probability or accuracy metric.")
            
        st.divider()
        st.subheader("Synthetic Demo Dataset Leaderboard")
        st.caption("The CSV rows are synthetically generated. This leaderboard illustrates inference only and is not evidence of real-world model performance.")
        
        if os.path.exists('test_candidates.csv'):
            df = pd.read_csv('test_candidates.csv')
            # The CSV's "shortlisted" column is a synthetic training label, not a real candidate outcome.
            if "shortlisted" in df.columns:
                df = df.drop(columns=["shortlisted"])

            # Predict for all
            X_test_raw = df[['years_experience', 'technical_score', 'soft_skills_score', 'projects_completed']].values
            X_norm, _, _ = min_max_normalize(X_test_raw, nn.min_vals, nn.max_vals)
            
            _, probs = nn.predict(X_norm)
            df['Fit Score (%)'] = (probs.flatten() * 100).round(2)
            df['Interpretation'] = "Demo score only — review manually"
            
            # Sort by Fit Score
            df_sorted = df.sort_values(by='Fit Score (%)', ascending=False).reset_index(drop=True)
            df_sorted.index = df_sorted.index + 1
            st.dataframe(df_sorted, hide_index=True, use_container_width=True)
        else:
            st.warning("test_candidates.csv not found! Run train.py first.")


if __name__ == '__main__':
    main()
