"""Integration tests against the trained checkpoint; no mocked predictions."""
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import torch
from fastapi.testclient import TestClient

from app import create_app


class InferenceUITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.app = create_app(device="cpu")
        cls.context = TestClient(cls.app)
        cls.client = cls.context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.context.__exit__(None, None, None)

    def test_page_and_assets(self):
        for path, content in [("/", 'lang="th"'), ("/static/app.js", "/api/predict"),
                              ("/static/style.css", "@media")]:
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200)
            self.assertIn(content, response.text)

    def test_real_prediction_and_single_model_instance(self):
        predictor = self.app.state.predictor
        review = "อาหารอร่อยมาก บริการดีเยี่ยม"
        response = self.client.post("/api/predict", json={"text": review})
        self.assertEqual(response.status_code, 200)
        result = response.json()
        probabilities = result["probabilities"]
        self.assertEqual(set(probabilities), {"1", "2", "3", "4", "5"})
        self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=5)
        self.assertTrue(all(0 <= value <= 1 for value in probabilities.values()))
        self.assertEqual(result["stars"], int(max(probabilities, key=probabilities.get)))
        self.assertEqual(result["stars"], predictor.predict([review])[0]["stars"])
        self.assertFalse(result["truncated"])
        self.assertIs(self.app.state.predictor, predictor)

    def test_invalid_requests(self):
        for body in [{}, {"text": ""}, {"text": " \n\t "}, {"text": "ก" * 20001}, {"text": 123}]:
            with self.subTest(body_type=type(body.get("text"))):
                self.assertEqual(self.client.post("/api/predict", json=body).status_code, 422)

    def test_long_review_discloses_truncation(self):
        response = self.client.post("/api/predict", json={"text": "อาหารอร่อย บริการดี " * 200})
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertTrue(result["truncated"])
        self.assertEqual(result["used_tokens"], 256)
        self.assertGreater(result["token_count"], 256)

    def test_status(self):
        result = self.client.get("/api/status").json()
        self.assertTrue(result["ready"])
        self.assertEqual(result["device"], "cpu")
        self.assertEqual(result["max_length"], 256)
        self.assertIsNotNone(result["evaluation"])

    def test_missing_checkpoint_still_serves_ui(self):
        with TemporaryDirectory() as directory:
            with self.assertLogs("app", level="ERROR"):
                with TestClient(create_app(Path(directory), device="cpu")) as client:
                    self.assertEqual(client.get("/").status_code, 200)
                    self.assertFalse(client.get("/api/status").json()["ready"])
                    self.assertEqual(client.post("/api/predict", json={"text": "อาหารอร่อย"}).status_code, 503)


if __name__ == "__main__":
    unittest.main()
