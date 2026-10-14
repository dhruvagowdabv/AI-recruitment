import unittest

from role_profiles import (
    analyze_role_match,
    build_role_description,
    get_role_profile,
)


class RoleProfilesTests(unittest.TestCase):
    def test_data_analyst_profile_has_required_and_preferred_skills(self):
        profile = get_role_profile("Data Analyst")
        self.assertIn("SQL", profile["required_skills"])
        self.assertIn("Tableau", profile["preferred_skills"])
        self.assertIn("Data Analyst", build_role_description(profile))

    def test_profile_mode_matches_aliases_case_insensitively(self):
        profile = get_role_profile("Data Analyst")
        result = analyze_role_match(
            resume_text="DATA ANALYST with Python, PostgreSQL, Excel and a Tableau dashboard project.",
            target_text=build_role_description(profile),
            model_result={"match_score": 0.7},
            use_profile=True,
        )
        self.assertIn("SQL", result["matched_skills"])
        self.assertIn("Python", result["matched_skills"])
        self.assertIn("Excel", result["matched_skills"])
        self.assertGreaterEqual(result["role_match_percentage"], 0)
        self.assertLessEqual(result["role_match_percentage"], 100)
        self.assertEqual(len(result["score_breakdown"]), 3)

    def test_custom_jd_focuses_skill_targets(self):
        result = analyze_role_match(
            resume_text="Python developer with SQL experience.",
            target_text="Data Analyst required: SQL and Python. Experience with Tableau preferred.",
            model_result={"match_score": 0.5},
            use_profile=False,
        )
        self.assertIn("SQL", result["matched_skills"])
        self.assertIn("Python", result["matched_skills"])
        self.assertIn("Tableau", result["missing_skills"])

    def test_empty_resume_returns_gaps_and_suggestions(self):
        result = analyze_role_match(
            resume_text="",
            target_text=build_role_description(get_role_profile("Data Analyst")),
            model_result={"match_score": 0.2},
            use_profile=True,
        )
        self.assertTrue(result["missing_skills"])
        self.assertTrue(result["improvement_suggestions"])

    def test_unsupported_role_fails_clearly(self):
        with self.assertRaises(ValueError):
            get_role_profile("Unlisted Role")


if __name__ == "__main__":
    unittest.main()
