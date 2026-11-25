"""Predict a 1-5 star rating using the saved fine-tuned checkpoint."""
import argparse
import json
from threading import Lock
from pathlib import Path
from common import ROOT, load_model, load_tokenizer
import torch


class ReviewPredictor:
    """Load a checkpoint once and serialize inference on its device."""

    def __init__(self, model_path=ROOT / "runs/wongnai/best_model", device="auto"):
        self.path = Path(model_path)
        settings = json.loads((self.path / "inference_config.json").read_text(encoding="utf-8"))
        self.max_length = int(settings["max_length"])
        self.best_epoch = settings.get("best_epoch")
        self.tokenizer = load_tokenizer(self.path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        self.model = load_model(self.path).eval().to(self.device)
        self.lock = Lock()

    def predict(self, texts):
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Please provide a non-empty review")
        with self.lock, torch.inference_mode():
            # Count the original tokens so the UI can disclose truncation accurately.
            full = self.tokenizer(texts, truncation=False)
            lengths = [len(ids) for ids in full["input_ids"]]
            inputs = self.tokenizer(texts, padding=True, truncation=True,
                                    max_length=self.max_length, return_tensors="pt").to(self.device)
            probabilities = self.model(**inputs).logits.float().softmax(-1).cpu()
        return [{"text": text, "stars": int(values.argmax()) + 1,
                 "probabilities": {str(i + 1): p for i, p in enumerate(values.tolist())},
                 "token_count": length, "used_tokens": min(length, self.max_length),
                 "truncated": length > self.max_length}
                for text, values, length in zip(texts, probabilities, lengths)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="+", help="One or more quoted Thai reviews")
    parser.add_argument("--model", default="runs/wongnai/best_model")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    predictor = ReviewPredictor(ROOT / args.model, device=args.device)
    for result in predictor.predict(args.text):
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
