"""Explainable, rule-based resume-to-role skill evidence scoring.

Scores describe textual evidence coverage only. They are not hiring probabilities,
candidate quality, or estimates of job performance.
"""
import re
from typing import Dict, List

ROLE_PROFILES = {
    "Data Analyst": {
        "description": (
            "Data Analyst role requiring SQL, Excel, Python, data cleaning, data analysis, "
            "data visualization and statistics. Preferred: Power BI, Tableau, Pandas, NumPy, "
            "exploratory data analysis, KPI reporting, dashboard development and business insights."
        ),
        "required_skills": [
            "SQL", "Excel", "Python", "Data Cleaning", "Data Analysis",
            "Data Visualization", "Statistics",
        ],
        "preferred_skills": [
            "Power BI", "Tableau", "Pandas", "NumPy", "Exploratory Data Analysis",
            "KPI Reporting", "Dashboard Development", "Business Insights",
        ],
    },
    "Data Engineer": {
        "description": (
            "Data Engineer role requiring Python, SQL, ETL, data pipelines and databases. "
            "Preferred: Apache Spark, Airflow, cloud platforms, Docker and data warehouses."
        ),
        "required_skills": ["Python", "SQL", "ETL", "Data Pipelines", "Databases"],
        "preferred_skills": ["Apache Spark", "Apache Airflow", "AWS", "Azure", "Docker", "Data Warehousing"],
    },
    "Machine Learning Engineer": {
        "description": (
            "Machine Learning Engineer role requiring Python, machine learning, model evaluation, "
            "data preprocessing and scikit-learn. Preferred: PyTorch, TensorFlow, Docker, APIs and MLOps."
        ),
        "required_skills": ["Python", "Machine Learning", "Model Evaluation", "Data Preprocessing", "scikit-learn"],
        "preferred_skills": ["PyTorch", "TensorFlow", "Docker", "FastAPI", "MLOps"],
    },
    "Software Engineer": {
        "description": (
            "Software Engineer role requiring Python or Java, data structures, algorithms, Git and testing. "
            "Preferred: REST APIs, SQL, Docker, CI/CD and cloud platforms."
        ),
        "required_skills": ["Python", "Java", "Data Structures", "Algorithms", "Git", "Testing"],
        "preferred_skills": ["REST API", "SQL", "Docker", "CI/CD", "AWS", "Azure"],
    },
}

SKILL_ALIASES = {
    "SQL": [r"\bsql\b", r"\bpostgres(?:ql)?\b", r"\bmysql\b", r"\bsql server\b", r"\bsqlite\b"],
    "Excel": [r"\bexcel\b", r"\bmicrosoft excel\b", r"\bspreadsheets?\b"],
    "Python": [r"\bpython\b"],
    "Java": [r"\bjava\b"],
    "Data Cleaning": [r"\bdata cleaning\b", r"\bdata cleansing\b", r"\bcleaning data\b"],
    "Data Analysis": [r"\bdata analysis\b", r"\bdata analytics\b", r"\banalysed data\b", r"\banalyzed data\b"],
    "Data Visualization": [r"\bdata visuali[sz]ation\b", r"\bvisuali[sz]ation\b", r"\bcharts?\b", r"\bplots?\b"],
    "Statistics": [r"\bstatistics\b", r"\bstatistical analysis\b", r"\bprobability\b"],
    "Power BI": [r"\bpower\s*bi\b"],
    "Tableau": [r"\btableau\b"],
    "Pandas": [r"\bpandas\b"],
    "NumPy": [r"\bnumpy\b"],
    "Exploratory Data Analysis": [r"\bexploratory data analysis\b", r"\beda\b"],
    "KPI Reporting": [r"\bkpis?\b", r"\bkey performance indicators?\b"],
    "Dashboard Development": [r"\bdashboards?\b", r"\bdashboard development\b"],
    "Business Insights": [r"\bbusiness insights?\b", r"\bactionable insights?\b", r"\bbusiness recommendations?\b"],
    "Matplotlib": [r"\bmatplotlib\b"],
    "R": [r"\br programming\b", r"\blanguage r\b"],
    "Looker": [r"\blooker\b"],
    "DAX": [r"\bdax\b"],
    "ETL": [r"\betl\b", r"\bextract(?:ion)?[,\s]+transform(?:ation)?[,\s]+load\b"],
    "Data Pipelines": [r"\bdata pipelines?\b", r"\bpipeline orchestration\b"],
    "Databases": [r"\bdatabases?\b", r"\bdatabase management\b", r"\brelational database\b"],
    "Apache Spark": [r"\bapache spark\b", r"\bspark\b", r"\bpyspark\b"],
    "Apache Airflow": [r"\bapache airflow\b", r"\bairflow\b"],
    "Azure": [r"\bazure\b"],
    "AWS": [r"\baws\b", r"\bamazon web services\b"],
    "Docker": [r"\bdocker\b"],
    "Data Warehousing": [r"\bdata warehouses?\b", r"\bdata warehousing\b", r"\bsnowflake\b", r"\bredshift\b"],
    "Machine Learning": [r"\bmachine learning\b", r"\bml\b"],
    "Model Evaluation": [r"\bmodel evaluation\b", r"\bevaluation metrics\b", r"\bprecision\b", r"\brecall\b", r"\bf1 score\b"],
    "Data Preprocessing": [r"\bdata preprocessing\b", r"\bpreprocessing\b", r"\bfeature engineering\b"],
    "scikit-learn": [r"\bscikit[\s-]?learn\b", r"\bsklearn\b"],
    "PyTorch": [r"\bpytorch\b"],
    "TensorFlow": [r"\btensorflow\b"],
    "FastAPI": [r"\bfastapi\b"],
    "MLOps": [r"\bmlops\b", r"\bmodel deployment\b", r"\bmodel monitoring\b"],
    "Data Structures": [r"\bdata structures?\b"],
    "Algorithms": [r"\balgorithms?\b", r"\bproblem solving\b"],
    "Git": [r"\bgit\b", r"\bgithub\b", r"\bversion control\b"],
    "Testing": [r"\btesting\b", r"\bunit tests?\b", r"\bpytest\b", r"\btest automation\b"],
    "REST API": [r"\brest(?:ful)? apis?\b", r"\bapi development\b"],
    "CI/CD": [r"\bci\s*/\s*cd\b", r"\bcontinuous integration\b", r"\bcontinuous delivery\b"],
}

NEGATION = re.compile(r"\b(?:no|not|without|lack(?:s|ing)?|never used|unfamiliar with)\s+(?:(?:any|prior|previous)\s+)?(?:experience with\s+|knowledge of\s+|proficiency in\s+)?$", re.I)


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
    return "{} role. Required skills: {}. Preferred skills: {}.".format(
        profile["name"], ", ".join(profile["required_skills"]), ", ".join(profile["preferred_skills"])
    )


def _contains_skill(text: str, skill: str) -> bool:
    text = text or ""
    patterns = SKILL_ALIASES.get(skill, [r"\b" + re.escape(skill) + r"\b"])
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.I):
            prefix = text[max(0, match.start() - 45):match.start()]
            if not NEGATION.search(prefix):
                return True
    return False


def _skills_mentioned(text: str, skills: List[str]) -> List[str]:
    return [skill for skill in dict.fromkeys(skills) if _contains_skill(text, skill)]


def recognized_skills(text: str) -> List[str]:
    """Return canonical skills with textual evidence in a string."""
    return _skills_mentioned(text or "", list(SKILL_ALIASES))


def _classify_custom_skills(target_text: str):
    """Classify recognized JD skills using local sentence cues.

    Unmarked requirements default to required. This is a lightweight heuristic;
    recruiters should verify the classification.
    """
    required, preferred = [], []
    text = target_text or ""
    preferred_cues = re.compile(
        r"\b(?:preferred|nice to have|bonus|desirable|ideally|optional|a plus|plus if|would be a plus)\b",
        re.I,
    )
    for skill in _skills_mentioned(text, list(SKILL_ALIASES)):
        for pattern in SKILL_ALIASES.get(skill, [r"\b" + re.escape(skill) + r"\b"]):
            match = re.search(pattern, text, re.I)
            if not match:
                continue
            starts = [text.rfind(mark, 0, match.start()) for mark in (".", "!", "?", ";", "\n")]
            start = max(starts) + 1
            ends = [text.find(mark, match.end()) for mark in (".", "!", "?", ";", "\n")]
            ends = [end for end in ends if end >= 0]
            end = min(ends) if ends else len(text)
            context = text[start:end]
            (preferred if preferred_cues.search(context) else required).append(skill)
            break
    return required, preferred


def analyze_role_match(
    resume_text: str,
    target_text: str,
    model_result: Dict = None,
    use_profile: bool = True,
    role_name: str = "Data Analyst",
) -> Dict:
    """Score target-skill evidence coverage without relying on an uncalibrated model.

    Predefined profiles weight required coverage at 70% and preferred coverage at
    30%. Custom JDs use nearby wording cues to classify skills as required or
    preferred, then use the same weighting when preferred skills are present. A custom
    JD with no recognized skills has no valid score and returns score_available=False.
    """
    # The neural model result is optional; normalize missing or malformed values safely.
    if not isinstance(model_result, dict):
        model_result = None

    profile = get_role_profile(role_name)
    resume_text = resume_text or ""
    target_text = target_text or ""

    if use_profile:
        required = profile["required_skills"]
        preferred = profile["preferred_skills"]
        target_skills = list(dict.fromkeys(required + preferred))
        matched_required = _skills_mentioned(resume_text, required)
        matched_preferred = _skills_mentioned(resume_text, preferred)
        required_coverage = len(matched_required) / len(required) if required else 0.0
        preferred_coverage = len(matched_preferred) / len(preferred) if preferred else 0.0
        score_available = True
        score = 0.70 * required_coverage + 0.30 * preferred_coverage
    else:
        required, preferred = _classify_custom_skills(target_text)
        target_skills = list(dict.fromkeys(required + preferred))
        matched_required = _skills_mentioned(resume_text, required)
        matched_preferred = _skills_mentioned(resume_text, preferred)
        required_coverage = len(matched_required) / len(required) if required else 0.0
        preferred_coverage = len(matched_preferred) / len(preferred) if preferred else 0.0
        score_available = bool(target_skills)
        if not score_available:
            score = None
        elif preferred:
            score = 0.70 * required_coverage + 0.30 * preferred_coverage
        else:
            score = required_coverage

    matched = list(dict.fromkeys(matched_required + matched_preferred))
    missing_required = [s for s in required if s not in matched_required]
    missing_preferred = [s for s in preferred if s not in matched_preferred]
    missing = missing_required + missing_preferred
    percentage = round(score * 100, 1) if score is not None else None
    coverage_percentage = round(len(matched) / len(target_skills) * 100, 1) if target_skills else None

    suggestions = []
    if missing_required:
        suggestions.append("If you have this experience, add specific evidence for required skills: " + ", ".join(missing_required) + ".")
    if missing_preferred:
        suggestions.append("Consider demonstrating preferred skills through a project or work example: " + ", ".join(missing_preferred) + ".")
    if not target_skills:
        suggestions.append("No supported skills were recognized in this job description. Add a clearer JD or extend the skill vocabulary before comparing candidates.")
    if not re.search(r"\b(project|dashboard|analysis|report|business insight|case study|deployed|built|implemented)\b", resume_text, re.I):
        suggestions.append("If applicable, add a concrete project or work example describing the problem, tools used and outcome.")
    if not suggestions:
        suggestions.append("Keep skill claims tied to specific projects, responsibilities and outcomes.")

    breakdown = []
    if use_profile:
        breakdown.extend([
            {"Component": "Required skill coverage (70% weight)", "Score (%)": round(required_coverage * 100, 1), "Contribution (points)": round(required_coverage * 70, 1)},
            {"Component": "Preferred skill coverage (30% weight)", "Score (%)": round(preferred_coverage * 100, 1), "Contribution (points)": round(preferred_coverage * 30, 1)},
        ])
    elif score_available:
        if preferred:
            breakdown.extend([
                {"Component": "Required JD skill coverage (70% weight)", "Score (%)": round(required_coverage * 100, 1), "Contribution (points)": round(required_coverage * 70, 1)},
                {"Component": "Preferred JD skill coverage (30% weight)", "Score (%)": round(preferred_coverage * 100, 1), "Contribution (points)": round(preferred_coverage * 30, 1)},
            ])
        else:
            breakdown.append({"Component": "Recognized JD skill coverage", "Score (%)": percentage, "Contribution (points)": percentage})
    else:
        breakdown.append({"Component": "Skill coverage", "Score (%)": "N/A", "Contribution (points)": "N/A"})
    breakdown.append({"Component": "Overall evidence coverage", "Score (%)": percentage if percentage is not None else "N/A", "Contribution (points)": percentage if percentage is not None else "N/A"})

    return {
        "role_name": role_name,
        "score_available": score_available,
        "role_match_score": round(score, 4) if score is not None else None,
        "role_match_percentage": percentage,
        "decision": "Review evidence" if score_available else "Score unavailable",
        "skill_coverage": round(len(matched) / len(target_skills), 4) if target_skills else None,
        "skill_coverage_percentage": coverage_percentage,
        "matched_skills": matched,
        "missing_skills": missing,
        "matched_required_skills": matched_required,
        "missing_required_skills": missing_required,
        "matched_preferred_skills": matched_preferred,
        "missing_preferred_skills": missing_preferred,
        "matched_skill_count": len(matched),
        "target_skill_count": len(target_skills),
        "score_breakdown": breakdown,
        "improvement_suggestions": suggestions,
        "scoring_mode": "Predefined role profile" if use_profile else "Custom job description",
        "required_skills": required,
        "preferred_skills": preferred,
        "score_note": "Textual evidence coverage only; not a probability of hiring or job performance.",
    }
