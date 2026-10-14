import unittest

from role_profiles import analyze_role_match, get_role_profile, build_role_description


class EvidenceMatchingTests(unittest.TestCase):
    def test_predefined_profile_score_uses_required_and_preferred_coverage(self):
        profile = get_role_profile("Data Analyst")
        result = analyze_role_match(
            "Python SQL Excel data cleaning data analysis data visualization statistics Tableau",
            build_role_description(profile),
            use_profile=True,
            role_name="Data Analyst",
        )
        self.assertTrue(result["score_available"])
        self.assertGreater(result["role_match_percentage"], 70)
        self.assertIn("SQL", result["matched_required_skills"])
        self.assertIn("Tableau", result["matched_preferred_skills"])

    def test_custom_jd_is_scored_only_on_recognized_skills(self):
        result = analyze_role_match(
            "Python and PostgreSQL developer",
            "We need SQL, Python and Tableau.",
            use_profile=False,
        )
        self.assertEqual(result["target_skill_count"], 3)
        self.assertEqual(result["matched_skill_count"], 2)
        self.assertAlmostEqual(result["role_match_percentage"], 66.7, places=1)

    def test_custom_jd_classifies_preferred_skills_from_local_wording(self):
        result = analyze_role_match(
            "Python developer",
            "Python is required. Tableau is preferred.",
            use_profile=False,
        )
        self.assertIn("Python", result["required_skills"])
        self.assertIn("Tableau", result["preferred_skills"])
        self.assertEqual(result["role_match_percentage"], 70.0)

    def test_custom_jd_without_supported_skills_has_no_score(self):
        result = analyze_role_match(
            "Strong communicator and self-starter",
            "Seeking a motivated self-starter with a positive attitude.",
            use_profile=False,
        )
        self.assertFalse(result["score_available"])
        self.assertIsNone(result["role_match_percentage"])

    def test_negated_skill_is_not_counted_as_positive_evidence(self):
        result = analyze_role_match(
            "No Python experience; familiar with SQL.",
            "Python and SQL required.",
            use_profile=False,
        )
        self.assertNotIn("Python", result["matched_skills"])
        self.assertIn("Python", result["missing_skills"])

    def test_aliases_do_not_match_substrings(self):
        result = analyze_role_match(
            "JavaScript developer with PostgreSQL",
            "Java and SQL required",
            use_profile=False,
        )
        self.assertNotIn("Java", result["matched_skills"])
        self.assertIn("SQL", result["matched_skills"])

    def test_multiple_profiles_are_available(self):
        for role in ("Data Analyst", "Data Engineer", "Machine Learning Engineer", "Software Engineer"):
            self.assertTrue(get_role_profile(role)["required_skills"])

    def test_model_result_does_not_affect_primary_score(self):
        args = dict(
            resume_text="Python SQL Excel Tableau",
            target_text=build_role_description(get_role_profile("Data Analyst")),
            use_profile=True,
        )
        a = analyze_role_match(**args, model_result={"match_score": 0.01})
        b = analyze_role_match(**args, model_result={"match_score": 0.99})
        self.assertEqual(a["role_match_percentage"], b["role_match_percentage"])


if __name__ == "__main__":
    unittest.main()
