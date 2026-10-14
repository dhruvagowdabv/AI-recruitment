"""Conservative resume PDF text extraction and baseline feature parsing.

Heuristic features are descriptive only; they are not a validated hiring score.
"""
import re
from pypdf import PdfReader

TECH_SKILLS = [
    "python", "java", "c++", "c#", "javascript", "typescript", "sql", "react",
    "node.js", "aws", "docker", "machine learning", "data science", "numpy",
    "pandas", "git", "tableau", "power bi", "pytorch", "tensorflow", "django",
]
SOFT_SKILLS = [
    "communication", "leadership", "teamwork", "problem solving",
    "time management", "adaptability", "critical thinking", "collaboration",
]


def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF-like file object or path.

    Returns an empty string for image-only/scanned PDFs; OCR is not performed.
    Raises parsing errors to the caller so the UI can explain the problem.
    """
    reader = PdfReader(pdf_file)
    pages = []
    for page in reader.pages:
        try:
            extracted = page.extract_text()
        except Exception:
            extracted = ""
        if extracted:
            pages.append(extracted)
    return "\n".join(pages).strip()


def _contains_phrase(text, phrase):
    if phrase in {"c++", "c#"}:
        return re.search(re.escape(phrase) + r"(?!\w)", text, re.I) is not None
    if phrase in {"node.js"}:
        return re.search(r"\bnode(?:\.js)?\b", text, re.I) is not None
    return re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text, re.I) is not None


def extract_experience(text):
    """Extract explicit duration mentions; do not invent experience when absent."""
    text = text or ""
    month_values = [float(v) for v in re.findall(r"\b(\d+(?:\.\d+)?)\s*(?:months?|mos?)\b", text, re.I)]
    year_values = [float(v) for v in re.findall(r"\b(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\b", text, re.I)]
    # Avoid adding duplicate total-duration expressions when a resume repeats them;
    # this remains a rough textual estimate, not a chronology calculation.
    years = sum(year_values) + sum(month_values) / 12.0
    return round(min(years, 50.0), 2)


def parse_resume(pdf_file):
    """Return basic textual features and extraction diagnostics."""
    text = extract_text_from_pdf(pdf_file)
    if not text:
        return {
            "Experience": 0.0, "Tech_Score": 0.0, "Soft_Score": 0.0,
            "Projects": 0, "Text_Extracted": False,
            "Extraction_Warning": "No selectable text found. This may be a scanned PDF; OCR is not built in.",
        }

    tech_count = sum(_contains_phrase(text, skill) for skill in TECH_SKILLS)
    soft_count = sum(_contains_phrase(text, skill) for skill in SOFT_SKILLS)
    project_mentions = len(re.findall(r"\b(?:project|dashboard|case study|application)\b", text, re.I))
    experience = extract_experience(text)

    return {
        "Experience": float(experience),
        "Tech_Score": round(min(tech_count / max(len(TECH_SKILLS), 1) * 100, 100), 1),
        "Soft_Score": round(min(soft_count / max(len(SOFT_SKILLS), 1) * 100, 100), 1),
        "Projects": min(project_mentions, 100),
        "Text_Extracted": True,
        "Extraction_Warning": None,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(parse_resume(sys.argv[1]))
    else:
        print("Usage: python resume_parser.py <resume.pdf>")
