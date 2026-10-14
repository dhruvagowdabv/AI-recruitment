"""Transparent resume-to-role skill coverage.

Scores describe detected textual evidence only. They are not model accuracy,
a probability of job success, or an automatic hiring recommendation.
"""
import re
from typing import Dict, List

ROLE_PROFILES = {
    "Data Analyst": {
        "description": (
            "Data Analyst role requiring SQL, Excel, Python, data cleaning, data "
            "analysis, statistics, and data visualization. Preferred skills include "
            "Power BI, Tableau, Pandas, NumPy, exploratory data analysis, KPI reporting, "
            "dashboard development, and communicating business insights."
        ),
        "required_skills": [
            "SQL", "Excel", "Python", "Data Cleaning", "Data Analysis",
            "Data Visualization", "Statistics",
        ],
        "preferred_skills": [
            "Power BI", "Tableau", "Pandas", "NumPy",
            "Exploratory Data Analysis", "KPI Reporting",
            "Dashboard Development", "Business Insights",
        ],
    }
}

# Canonical skill -> explicit phrases that count as textual evidence.
# Avoid generic words such as "reporting" on their own: they create false matches.
SKILL_ALIASES = {
    "SQL": [r"\bsql\b", r"\bpostgres(?:ql)?\b", r"\bmysql\b", r"\bsql server\b"],
    "Excel": [r"\bexcel\b", r"\bmicrosoft excel\b", r"\bspreadsheets?\b"],
    "Python": [r"\bpython\b"],
    "Data Cleaning": [r"\bdata cleaning\b", r"\bdata cleansing\b", r"\bcleaning data\b"],
    "Data Analysis": [r"\bdata analysis\b", r"\bdata analytics\b"],
    "Data Visualization": [r"\bdata visuali[sz]ation\b", r"\bvisuali[sz]ation of data\b"],
    "Statistics": [r"\bstatistics\b", r"\bstatistical analysis\b"],
    "Power BI": [r"\bpower\s*bi\b"],
    "Tableau": [r"\btableau\b"],
    "Pandas": [r"\bpandas\b"],
    "NumPy": [r"\bnumpy\b"],
    "Exploratory Data Analysis": [r"\bexploratory data analysis\b", r"\beda\b"],
    "KPI Reporting": [r"\bkpis?\b", r"\bkey performance indicators?\b"],
    "Dashboard Development": [r"\bdashboards?\b"],
    "Business Insights": [r"\bbusiness insights?\b", r"\bactionable insights?\b", r"\bbusiness recommendations?\b"],
    "Matplotlib": [r"\bmatplotlib\b"],
    "R": [r"\br programming\b", r"\blanguage r\b"],
    "Looker": [r"\blooker\b"],
    "DAX": [r"\bdax\b"],
    "ETL": [r"\betl\b", r"\bextract[ ,]+transform[ ,]+load\b"],
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
    return "{} role. Required skills: {}. Preferred skills: {}.".format(
        profile["name"],
        ", ".join(profile["required_skills"]),
        ", ".join(profile["preferred_skills"]),
    )


def _contains_skill(text: str, skill: str) -> bool:
    """Check explicit skill aliases with case-insensitive word boundaries."""
    normalized = text or ""
    patterns = SKILL_ALIASES.get(skill, [r"\b" + re.escape(skill) + r"\b"])
    return any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in patterns)


def _skills_mentioned(text: str, skills: List[str]) -> List[str]:
    return [skill for skill in skills if _contains_skill(text, skill)]


def analyze_role_match(
    resume_text: str,
    target_text: str,
    model_result: Dict = None,
    use_profile: bool = True,
    role_name: str = "Data Analyst",
) -> Dict:
    """Score explicit skill evidence without blending in an unvalidated neural score.

    For the predefined profile, required-skill coverage contributes 70% and
    preferred-skill coverage contributes 30%. For a custom JD, recognized skills
    mentioned in the JD are treated as target skills and coverage is reported
    directly. A missing skill means it was not found in text, not that the person
    lacks the skill.
    """
    profile = get_role_profile(role_name)
    resume = resume_text or ""
    target = target_text or ""

    if use_profile:
        required = profile["required_skills"]
        preferred = profile["preferred_skills"]
        target_skills = list(dict.fromkeys(required + preferred))
        matched = _skills_mentioned(resume, target_skills)
        missing = [skill for skill in target_skills if skill not in matched]
        required_matched = [s for s in required if s in matched]
        preferred_matched = [s for s in preferred if s in matched]
        required_coverage = len(required_matched) / len(required) if required else 0.0
        preferred_coverage = len(preferred_matched) / len(preferred) if preferred else 0.0
        role_score = 0.70 * required_coverage + 0.30 * preferred_coverage
        breakdown = [
            {"Component": "Required-skill evidence (70%)", "Score (%)": round(required_coverage * 100, 1), "Contribution (points)": round(required_coverage * 70, 1)},
            {"Component": "Preferred-skill evidence (30%)", "Score (%)": round(preferred_coverage * 100, 1), "Contribution (points)": round(preferred_coverage * 30, 1)},
            {"Component": "Overall evidence coverage", "Score (%)": round(role_score * 100, 1), "Contribution (points)": round(role_score * 100, 1)},
        ]
    else:
        recognized = list(SKILL_ALIASES.keys())
        target_skills = _skills_mentioned(target, recognized)
        matched = _skills_mentioned(resume, target_skills)
        missing = [skill for skill in target_skills if skill not in matched]
        required = target_skills
        preferred = []
        required_matched = matched
        preferred_matched = []
        required_coverage = len(matched) / len(target_skills) if target_skills else 0.0
        preferred_coverage = 0.0
        role_score = required_coverage if target_skills else None
        breakdown = (
            [{"Component": "Recognized skills named in custom JD", "Score (%)": round(role_score * 100, 1), "Contribution (points)": round(role_score * 100, 1)}]
            if target_skills else
            [{"Component": "Skill coverage", "Score (%)": "N/A", "Contribution (points)": "N/A"}]
        )

    suggestions = []
    missing_required = [skill for skill in required if skill in missing]
    missing_preferred = [skill for skill in preferred if skill in missing]
    if missing_required:
        suggestions.append(
            "If you have this experience, make the evidence easier to find for: "
            + ", ".join(missing_required) + "."
        )
    if missing_preferred:
        suggestions.append(
            "If applicable, add a project or practical example demonstrating: "
            + ", ".join(missing_preferred) + "."
        )
    if not re.search(r"\b(project|dashboard|analysis|report|business insight|case study)\b", resume, re.I):
        suggestions.append(
            "Consider describing one relevant project with the problem, methods, and outcome, if you have one."
        )
    if not suggestions:
        suggestions.append(
            "Keep skill claims specific by linking them to projects, tasks, or outcomes where possible."
        )

    score_available = role_score is not None
    percentage = round(role_score * 100, 1) if score_available else None
    return {
        "role_name": role_name,
        "role_match_score": round(role_score, 4) if score_available else None,
        "role_match_percentage": percentage,
        "score_available": score_available,
        "decision": "Review evidence; this score is not a hiring decision",
        "skill_coverage": round(role_score, 4) if score_available else None,
        "skill_coverage_percentage": percentage,
        "required_skill_coverage_percentage": round(required_coverage * 100, 1),
        "preferred_skill_coverage_percentage": round(preferred_coverage * 100, 1),
        "matched_skills": matched,
        "missing_skills": missing,
        "matched_skill_count": len(matched),
        "target_skill_count": len(target_skills),
        "score_breakdown": breakdown,
        "improvement_suggestions": suggestions,
        "scoring_mode": "Predefined role profile" if use_profile else "Custom job description",
        "score_note": (
            "Percentage of configured skill evidence found in the resume text. "
            "Not model accuracy, hiring probability, or proof of skill proficiency."
        ),
    }
