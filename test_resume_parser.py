import unittest

from resume_parser import _contains_phrase, extract_experience


class ResumeParserTests(unittest.TestCase):
    def test_experience_is_zero_when_no_duration_is_stated(self):
        self.assertEqual(extract_experience("Python developer with project experience"), 0.0)

    def test_explicit_years_and_months_are_converted(self):
        self.assertAlmostEqual(extract_experience("2 years and 6 months experience"), 2.5)

    def test_javascript_does_not_count_as_java(self):
        self.assertFalse(_contains_phrase("javascript developer", "java"))
        self.assertTrue(_contains_phrase("Java developer", "java"))

    def test_skill_matching_is_case_insensitive(self):
        self.assertTrue(_contains_phrase("Built with PYTHON and SQL", "python"))


if __name__ == "__main__":
    unittest.main()
