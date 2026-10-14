import unittest

from inference_pipeline import CandidateMatchingPipeline


class InferenceSafetyTests(unittest.TestCase):
    def test_empty_resume_is_rejected_before_inference(self):
        pipeline = CandidateMatchingPipeline()
        with self.assertRaises(ValueError):
            pipeline.evaluate_candidate("", "Python developer")

    def test_empty_job_description_is_rejected_before_inference(self):
        pipeline = CandidateMatchingPipeline()
        with self.assertRaises(ValueError):
            pipeline.evaluate_candidate("Python developer", " ")

    def test_unloaded_pipeline_does_not_return_a_fallback_score(self):
        pipeline = CandidateMatchingPipeline()
        with self.assertRaises(RuntimeError):
            pipeline.evaluate_candidate("Python developer", "Python required")

    def test_missing_siamese_model_does_not_return_neutral_similarity(self):
        pipeline = CandidateMatchingPipeline()
        with self.assertRaises(RuntimeError):
            pipeline.compute_resume_jd_similarity("resume", "job description")


if __name__ == "__main__":
    unittest.main()
