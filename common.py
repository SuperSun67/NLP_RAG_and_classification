"""Local-only model loading and shared evaluation utilities."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, mean_absolute_error
from transformers import CamembertTokenizer

LABELS = {i: str(i + 1) for i in range(5)}


def load_model(path):
    # Only code downloaded at the pinned revision and reviewed locally is used.
    sys.path.insert(0, str(ROOT / "assets" / "iii-bert"))
    from iii_modeling import IIIBERTForSequenceClassification
    model, info = IIIBERTForSequenceClassification.from_pretrained(
        str(path), num_labels=5, id2label=LABELS,
        label2id={v: k for k, v in LABELS.items()},
        local_files_only=True, output_loading_info=True,
    )
    missing = [k for k in info.get("missing_keys", []) if not k.startswith("classifier.")]
    if missing or info.get("mismatched_keys"):
        raise RuntimeError(f"Pretrained backbone did not load correctly: {info}")
    return model


def load_tokenizer(path):
    tokenizer = CamembertTokenizer.from_pretrained(str(path), local_files_only=True)
    # Camembert's fairseq-compatible vocabulary has gaps; len(tokenizer) is not
    # the embedding-table size. Check the actual IDs instead.
    if max(tokenizer.get_vocab().values()) >= 24008 or tokenizer.pad_token_id != 5 or tokenizer.eos_token_id != 6 or tokenizer.bos_token_id != 24005:
        raise RuntimeError(f"Tokenizer IDs do not match pretrained weights: {len(tokenizer)}, {tokenizer.special_tokens_map}")
    return tokenizer


def metrics(labels, predictions):
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, labels=list(LABELS), average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, predictions, labels=list(LABELS), average="weighted", zero_division=0),
        "mae_stars": mean_absolute_error(labels, predictions),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=list(LABELS)).tolist(),
        "classification_report": classification_report(labels, predictions, labels=list(LABELS), target_names=list(LABELS.values()), output_dict=True, zero_division=0),
    }
