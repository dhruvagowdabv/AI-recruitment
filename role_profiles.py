"""Configurable job-role profiles and transparent resume-to-role analysis.

The role-aware score is an explainable heuristic, not a calibrated hiring probability.
"""

import re
from typing import Dict, List, Optional


# Each entry contains a display name, required skills, preferred skills and a short
# description used as the comparison target when the user does not paste a custom JD.
ROLE_PROFILES = {
    "Data Analyst": {
        "description": (
            "Data Analyst role requiring SQL, Excel, Python, data cleaning, data "
            "analysis, statistics, and data visualization. Preferred skills include "
            "Power BI, Tableau, Pandas, NumPy, exploratory data analysis, KPI reporting, "
            "dashboard development, and communicating business insights."
        ),
        "required_skills": [
            "SQL",
            "Excel",
            "Python",
            "Data Cleaning",
            "Data Analysis",
            "Data Visualization",
            "Statistics",
        ],
        "preferred_skills": [
            "Power BI",
            "Tableau",
            "Pandas",
            "NumPy",
            "Exploratory Data Analysis",
            "KPI Reporting",
            "Dashboard Development",
            "Business Insights",
        ],
    }
}

# Canonical display name -> phrases accepted as evidence in resume/JD text.
SKILL_ALIASES = {
    "SQL": [r"\bsql\b", r"\bpostgres(?:ql)?\b", r"\bmysql\b", r"\bsql server\b"],
    "Excel": [r"\bexcel\b", r"\bmicrosoft excel\b", r"\bspreadsheet(?:s)?\b"],
    "Python": [r"\bpython\b"],
    "Data Cleaning": [r"\bdata cleaning\b", r"\bdata cleansing\b", r"\bcleaning data\b"],
    "Data Analysis": [r"\bdata analysis\b", r"\bdata analytics\b", r"\banalytical analysis\b"],
    "Data Visualization": [r"\bdata visualization\b", r"\bdata visualisation\b", r"\bvisualization\b", r"\bvisualisation\b"],
    "Statistics": [r"\bstatistics\b", r"\bstatistical analysis\b", r"\bprobability\b"],
    "Power BI": [r"\bpower\s*bi\b"],
    "Tableau": [r"\btableau\b"],
    "Pandas": [r"\bpandas\b"],
    "NumPy": [r"\bnumpy\b"],
    "Exploratory Data Analysis": [r"\bexploratory data analysis\b", r"\beda\b"],
    "KPI Reporting": [r"\bkpi(?:s)?\b", r"\bkey performance indicators?\b", r"\breporting\b"],
    "Dashboard Development": [r"\bdashboard(?:s)?\b", r"\bdashboard development\b"],
    "Business Insights": [r"\bbusiness insights?\b", r"\bactionable insights?\b", r"\bbusiness recommendations?\b"],
    "Matplotlib": [r"\bmatplotlib\b"],
    "NumPy": [r"\bnumpy\b"],
    "R": [r"\br programming\b", r"\blanguage r\b"],
    "Looker": [r"\blooker\b"],
    "DAX": [r"\bdax\b"],
    "ETL": [r"\betl\b", r"\bextract transform load\b"],
}


def get_role_profile(role_name: str) -> Dict:
    """Return a defensive copy of a supported role profile."""
    if role_name not in ROLE_PROFILES:
        raise ValueError("Unsupported role: {!r}".format(role_name))
    profile = ROLE_PROFILES[role_name]
    return {
        "name": role_name,
        "description": profile["description"],
        "required_skills": list(profile["required_skills"]),
        "preferred_skills": list(profile["preferred_skills"]),
    }


def build_role_description(profile: Dict) -> str:
    """Build a natural-language target from a role profile."""
    required = ", ".join(profile["required_skills"])
    preferred = ", ".join(profile["preferred_skills"])
    return "{} role. Required skills: {}. Preferred skills: {}.".format(
        profile["name"], required, preferred
    )


def _contains_skill(text: str, skill: str) -> bool:
    """Check canonical skill aliases using case-insensitive word boundaries."""
    normalized = text or ""
    patterns = SKILL_ALIASES.get(skill, [r"\b" + re.escape(skill.lower()) + r"\b"])
    return any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in patterns)


def _skills_mentioned(text: str, skills: List[str]) -> List[str]:
    return [skill for skill in skills if _contains_skill(text, skill)]


def analyze_role_match(
    resume_text: str,
    target_text: str,
    model_result: Dict,
    use_profile: bool = True,
    role_name: str = "Data Analyst",
) -> Dict:
    """Combine the existing neural score with transparent target-skill coverage.

    If a custom job description mentions recognized skills, those skills become the
    scoring targets. Otherwise, the predefined profile supplies required/preferred
    skills. Required skills carry more weight in profile mode.
    """
    profile = get_role_profile(role_name)
    resume = (resume_text or "").lower()
    target = (target_text or "").lower()

    required = list(profile["required_skills"])
    preferred = list(profile["preferred_skills"])
    target_skills = _skills_mentioned(target, list(SKILL_ALIASES.keys()))

    if not use_profile and target_skills:
        # A custom vacancy should focus the skill comparison on skills it actually names.
        required_skills = target_skills
        preferred_skills = []
        weights = {skill: 1.0 for skill in target_skills}
    else:
        required_skills = required
        preferred_skills = preferred
        weights = {skill: 2.0 for skill in required_skills}
        weights.update({skill: 1.0 for skill in preferred_skills})

    all_target_skills = list(dict.fromkeys(required_skills + preferred_skills))
    matched = _skills_mentioned(resume, all_target_skills)
    missing = [skill for skill in all_target_skills if skill not in matched]

    total_weight = sum(weights.get(skill, 1.0) for skill in all_target_skills)
    matched_weight = sum(weights.get(skill, 1.0) for skill in matched)
    skill_coverage = (matched_weight / total_weight) if total_weight else 0.0

    raw_model_score = model_result.get("match_score", 0.0)
    try:
        model_score = max(0.0, min(1.0, float(raw_model_score)))
    except (TypeError, ValueError):
        model_score = 0.0

    role_score = (0.60 * model_score) + (0.40 * skill_coverage)
    role_percentage = round(role_score * 100.0, 1)
    coverage_percentage = round(skill_coverage * 100.0, 1)

    suggestions = []
    if missing:
        required_missing = [skill for skill in required_skills if skill in missing]
        preferred_missing = [skill for skill in preferred_skills if skill in missing]
        if required_missing:
            suggestions.append(
                "If you have experience with them, add evidence of these core skills: "
                + ", ".join(required_missing) + "."
            )
        if preferred_missing:
            suggestions.append(
                "Consider adding a project or practical example demonstrating: "
                + ", ".join(preferred_missing) + "."
            )
    if not _contains_skill(resume, "Data Visualization") and not any(
        _contains_skill(resume, tool) for tool in ("Power BI", "Tableau")
    ):
        suggestions.append(
            "Add a data visualization or dashboard project with the tools used and the insights delivered, if applicable."
        )
    if not re.search(r"\b(project|dashboard|analysis|report|business insight|case study)\b", resume):
        suggestions.append(
            "Include one concrete analytics project describing the problem, methods, and measurable outcome, if you have one."
        )
    if not suggestions:
        suggestions.append(
            "Keep the resume evidence specific: connect relevant skills to projects, outcomes, and measurable impact."
        )

    return {
        "role_name": role_name,
        "role_match_score": round(role_score, 4),
        "role_match_percentage": role_percentage,
        "decision": "STRONGER ROLE MATCH" if role_percentage >= 50 else "ROLE GAPS IDENTIFIED",
        "skill_coverage": round(skill_coverage, 4),
        "skill_coverage_percentage": coverage_percentage,
        "matched_skills": matched,
        "missing_skills": missing,
        "matched_skill_count": len(matched),
        "target_skill_count": len(all_target_skills),
        "score_breakdown": [
            {"Component": "Existing neural ranking score (60%)", "Score (%)": round(model_score * 100, 1), "Contribution (points)": round(model_score * 60, 1)},
            {"Component": "Weighted target-skill coverage (40%)", "Score (%)": coverage_percentage, "Contribution (points)": round(skill_coverage * 40, 1)},
            {"Component": "Overall role match score", "Score (%)": role_percentage, "Contribution (points)": role_percentage},
        ],
        "improvement_suggestions": suggestions,
        "scoring_mode": "Predefined role profile" if use_profile else "Custom job description",
    }
