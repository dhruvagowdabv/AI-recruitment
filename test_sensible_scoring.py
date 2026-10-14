import unittest
from unittest.mock import patch

from resume_parser import extract_experience, parse_resume
from role_profiles import analyze_role_match, build_role_description, get_role_profile


class SensibleScoringTests(unittest.TestCase):
    def test_main_score_does_not_change_with_unvalidated_neural_output(self):
        resume = "Python SQL Excel data cleaning data analysis statistics Tableau dashboard project"
        profile = get_role_profile("Data Analyst")
        target = build_role_description(profile)
        low = analyze_role_match(resume, target, {"match_score": 0.05}, use_profile=True)
        high = analyze_role_match(resume, target, {"match_score": 0.99}, use_profile=True)
        self.assertEqual(low["role_match_percentage"], high["role_match_percentage"])
        self.assertTrue(low["score_available"])
        self.assertGreater(low["role_match_percentage"], 50)

    def test_profile_weights_required_skills_more_than_preferred(self):
        profile = get_role_profile("Data Analyst")
        target = build_role_description(profile)
        resume = "SQL Excel Python data cleaning data analysis data visualization statistics"
        result = analyze_role_match(resume, target, {}, use_profile=True)
        self.assertEqual(result["required_skill_coverage_percentage"], 100.0)
        self.assertEqual(result["preferred_skill_coverage_percentage"], 0.0)
        self.assertEqual(result["role_match_percentage"], 70.0)

    def test_custom_jd_uses_only_skills_it_names(self):
        result = analyze_role_match("Experienced analyst using PostgreSQL and Python.", "Must have SQL and Python. Tableau is preferred.", {}, use_profile=False)
        self.assertEqual(result["target_skill_count"], 3)
        self.assertEqual(result["matched_skill_count"], 2)
        self.assertAlmostEqual(result["role_match_percentage"], 66.7, places=1)

    def test_custom_jd_without_recognized_skills_has_no_score(self):
        result = analyze_role_match("Experienced analyst with strong reporting experience.", "Responsible for collaborating with stakeholders and preparing reports.", {}, use_profile=False)
        self.assertFalse(result["score_available"])
        self.assertIsNone(result["role_match_percentage"])
        self.assertEqual(result["target_skill_count"], 0)

    def test_generic_reporting_word_does_not_claim_kpi_skill(self):
        result = analyze_role_match("Prepared weekly reporting for a team.", "KPI reporting and dashboard development required.", {}, use_profile=False)
        self.assertNotIn("KPI Reporting", result["matched_skills"])

    def test_absent_experience_is_zero_not_fabricated(self):
        self.assertEqual(extract_experience("Python developer and data analyst"), 0.0)

    def test_parser_does_not_invent_missing_skills_or_projects(self):
        with patch("resume_parser.extract_text_from_pdf", return_value="javascript developer"):
            features = parse_resume(object())
        self.assertEqual(features["Experience"], 0.0)
        self.assertEqual(features["Tech_Score"], 10.0)
        self.assertEqual(features["Soft_Score"], 0.0)
        self.assertEqual(features["Projects"], 0)
        self.assertTrue(features["Text_Extracted"])


if __name__ == "__main__":
    unittest.main()
