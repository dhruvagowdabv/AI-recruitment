import re
from pypdf import PdfReader

TECH_SKILLS = ["python", "java", "c++", "c#", "javascript", "sql", "react", "node", "aws", "docker", "machine learning", "data science", "numpy", "pandas", "git"]
SOFT_SKILLS = ["communication", "leadership", "teamwork", "problem solving", "time management", "adaptability", "critical thinking", "agile", "collaboration", "management"]

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + " "
    return text.lower()

def extract_experience(text):
    """Estimate explicitly stated experience; zero means no duration was found."""
    text_lower = text.lower()
    
    # 1. Look for explicit months (e.g., "1 month", "2 months")
    month_matches = re.findall(r'(\d+)\s*(?:months?|mos?)', text_lower)
    total_months = sum([float(m) for m in month_matches])
    
    # 2. Look for explicit years (e.g., "2 years", "1.5 yrs")
    year_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:years?|yrs?)', text_lower)
    total_years = sum([float(y) for y in year_matches])
    
    # Combine (convert months to fractional years)
    calculated_exp = total_years + (total_months / 12.0)
    
    # Do not invent experience when the resume does not state a duration.
    return round(calculated_exp, 2) if calculated_exp > 0 else 0.0

def parse_resume(pdf_file):
    """
    Parses a resume PDF and extracts:
    - Experience (Years)
    - Tech Score (0-100)
    - Soft Score (0-100)
    - Projects (Count)
    Returns a dictionary of these features.
    """
    text = extract_text_from_pdf(pdf_file)
    if not text.strip():
        return {
            "Experience": 0.0,
            "Tech_Score": 0.0,
            "Soft_Score": 0.0,
            "Projects": 0,
            "Text_Extracted": False,
        }

    # 1. Experience Extraction
    exp = min(extract_experience(text), 10.0)
        
    # 2. Tech Score Extraction
    tech_count = 0
    for skill in TECH_SKILLS:
        # Avoid substring false positives such as "java" inside "javascript".
        pattern = r"(?<![a-z0-9])" + re.escape(skill) + r"(?![a-z0-9])"
        if re.search(pattern, text):
            tech_count += 1
    # This is a simple keyword count, not a proficiency measurement.
    tech_score = min(tech_count * 10, 100)
        
    # 3. Soft Skills Extraction
    soft_count = 0
    for skill in SOFT_SKILLS:
        if skill in text:
            soft_count += 1
    soft_score = min(soft_count * 15, 100)
        
    # 4. Projects Extraction
    # Count occurrences of words like "project", "developed", "built", "created"
    proj_keywords = ["project", "developed", "built", "created", "designed", "led"]
    proj_count = 0
    for word in proj_keywords:
        proj_count += len(re.findall(r'\b' + word + r'\b', text))
    
    # Estimate projects based on keywords (roughly divide by 2)
    projects = min(proj_count // 2, 6)
        
    return {
        "Experience": float(exp),
        "Tech_Score": float(tech_score),
        "Soft_Score": float(soft_score),
        "Projects": int(projects),
        "Text_Extracted": True,
    }

if __name__ == '__main__':
    # Simple test (assuming you have a test.pdf)
    import sys
    if len(sys.argv) > 1:
        print(parse_resume(sys.argv[1]))
    else:
        print("Run with a PDF file to test parsing: python resume_parser.py <file.pdf>")
