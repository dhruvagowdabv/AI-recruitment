import unittest

from evaluate_matching import evaluate_rows


class EvaluationTests(unittest.TestCase):
    def test_evaluation_reports_ranking_metrics_for_labeled_pairs(self):
        rows = [
            {"resume_text": "Python SQL Tableau", "job_description": "Need Python SQL Tableau", "label": "1"},
            {"resume_text": "Python SQL", "job_description": "Need Python SQL Tableau", "label": "1"},
            {"resume_text": "React frontend", "job_description": "Need Python SQL Tableau", "label": "0"},
        ]
        report = evaluate_rows(rows, k=2)
        self.assertEqual(report["examples"], 3)
        self.assertEqual(len(report["jobs"]), 1)
        self.assertIsNotNone(report["overall"]["roc_auc"])
        self.assertIsNotNone(report["jobs"][0]["precision_at_k"])

    def test_invalid_label_is_rejected(self):
        rows = [{"resume_text": "Python", "job_description": "Need Python", "label": "yes"}]
        with self.assertRaises(ValueError):
            evaluate_rows(rows)

    def test_unrecognized_job_description_is_rejected(self):
        rows = [{"resume_text": "Python", "job_description": "Nice attitude only", "label": "1"}]
        with self.assertRaises(ValueError):
            evaluate_rows(rows)


if __name__ == "__main__":
    unittest.main()
