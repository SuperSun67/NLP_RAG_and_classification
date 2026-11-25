"""Fine-tune III-BERT on Wongnai ratings using an offline, reproducible split."""
import argparse
import json
import math
import random
import time
import traceback
from collections import Counter
from pathlib import Path

from common import ROOT, load_model, load_tokenizer, metrics
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.nn import functional as F
from transformers import DataCollatorWithPadding, get_linear_schedule_with_warmup


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def prepare_data(seed):
    train = pd.read_parquet(ROOT / "assets/wongnai/data/train-00000-of-00001.parquet")
    test = pd.read_parquet(ROOT / "assets/wongnai/data/test-00000-of-00001.parquet")
    info = {"original_train": len(train), "original_test": len(test)}
    for frame in (train, test):
        if frame.review_body.isna().any() or not frame.star_rating.isin(range(5)).all():
            raise ValueError("Invalid review or rating")
        frame["text"] = frame.review_body.str.strip()
        if frame.text.eq("").any():
            raise ValueError("Empty review")
    # Keep the published test set intact. Text only, never test labels, is used for overlap removal.
    overlap = train.text.isin(set(test.text))
    info["train_test_overlap_removed"] = int(overlap.sum())
    train = train.loc[~overlap].copy()
    conflicting = train.groupby("text").star_rating.nunique()
    conflict_mask = train.text.isin(conflicting[conflicting > 1].index)
    info["conflicting_train_rows_removed"] = int(conflict_mask.sum())
    train = train.loc[~conflict_mask]
    before = len(train)
    train = train.drop_duplicates("text")
    info["duplicate_train_rows_removed"] = before - len(train)
    train, validation = train_test_split(train, test_size=0.1, stratify=train.star_rating, random_state=seed)
    frames = {"train": train, "validation": validation, "test": test}
    info["splits"] = {name: {"rows": len(frame), "stars": {str(k + 1): int(v) for k, v in sorted(Counter(frame.star_rating).items())}} for name, frame in frames.items()}
    info["source_row_indices"] = {name: frame.index.tolist() for name, frame in frames.items()}
    return frames, info


def encode(frame, tokenizer, max_length):
    encoded = tokenizer(frame.text.tolist(), truncation=True, max_length=max_length)
    return [{**{key: value[i] for key, value in encoded.items()}, "labels": int(label)} for i, label in enumerate(frame.star_rating)]


def get_class_weights(labels, mode="balanced"):
    """Inverse-frequency weights from the actual training subset only."""
    if mode == "none":
        return None
    if mode != "balanced":
        raise ValueError(f"Unknown class weighting mode: {mode}")
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=5)
    if len(counts) != 5 or np.any(counts == 0):
        raise ValueError("Balanced weighting requires all five classes in training")
    return torch.tensor(counts.sum() / (5 * counts), dtype=torch.float32)


def classification_loss_sum(logits, labels, class_weights=None):
    """Return weighted NLL sum and denominator for exact gradient accumulation."""
    loss_sum = F.cross_entropy(logits.float(), labels, weight=class_weights, reduction="sum")
    weight_sum = labels.numel() if class_weights is None else class_weights[labels].sum().item()
    return loss_sum, weight_sum


def normalize_gradients(parameters, weight_sum):
    # Normalize across the whole effective batch, not each microbatch separately.
    # This matches CrossEntropyLoss(weight=..., reduction="mean") on that batch.
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.div_(weight_sum)


@torch.inference_mode()
def evaluate(model, loader, device, use_bf16):
    model.eval()
    labels, predictions, probabilities = [], [], []
    loss_sum = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            output = model(**batch)
        loss_sum += output.loss.item() * len(batch["labels"])
        labels.extend(batch["labels"].cpu().tolist())
        predictions.extend(output.logits.argmax(-1).cpu().tolist())
        probabilities.extend(output.logits.float().softmax(-1).cpu().tolist())
    result = metrics(labels, predictions)
    result["loss"] = loss_sum / len(labels)
    return result, predictions, probabilities


def run(args, out):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA is unavailable; activate the project's CUDA environment")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    print(f"Device: {device}; bf16: {use_bf16}", flush=True)
    tokenizer = load_tokenizer(ROOT / "assets/iii-bert")
    model = load_model(ROOT / "assets/iii-bert").to(device)
    frames, data_info = prepare_data(args.seed)
    print(json.dumps({k: v for k, v in data_info.items() if k != "source_row_indices"}), flush=True)
    if args.smoke:
        frames = {name: frame.groupby("star_rating", group_keys=False).head(4) for name, frame in frames.items()}
    class_weights = get_class_weights(frames["train"].star_rating, args.class_weight)
    weighting = {
        "mode": args.class_weight,
        "source": "training subset only (after cleaning, splitting and optional smoke sampling)",
        "counts_by_star": {str(i + 1): int((frames["train"].star_rating == i).sum()) for i in range(5)},
        "weights_by_star": None if class_weights is None else {str(i + 1): float(w) for i, w in enumerate(class_weights)},
        "training_loss": "sum(weight[label] * NLL) / sum(weight[label]) over each accumulation window",
        "evaluation_loss": "unweighted mean cross entropy",
    }
    data_info["class_weighting"] = weighting
    write_json(out / "data_report.json", data_info)
    print("Class weighting: " + json.dumps(weighting), flush=True)
    if class_weights is not None:
        class_weights = class_weights.to(device)
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    loaders = {}
    for name, frame in frames.items():
        print(f"Tokenizing {name}: {len(frame)} reviews", flush=True)
        loaders[name] = DataLoader(encode(frame, tokenizer, args.max_length), batch_size=args.batch_size,
                                  shuffle=name == "train", collate_fn=collator, num_workers=0)
    updates_per_epoch = math.ceil(len(loaders["train"]) / args.accumulation)
    total_updates = updates_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, int(total_updates * 0.1)), total_updates)
    manifest = json.loads((ROOT / "assets/manifest.json").read_text())
    write_json(out / "run_config.json", {**vars(args), **manifest, "device": str(device),
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "torch": torch.__version__, "bf16": use_bf16, "effective_batch_size": args.batch_size * args.accumulation,
        "class_weighting": weighting})
    best_score = -1.0
    best_epoch = 0
    history = []
    global_step = 0
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        epoch_weight = 0.0
        window_weight = 0.0
        for step, batch in enumerate(loaders["train"], 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                output = model(**{k: v for k, v in batch.items() if k != "labels"})
                loss, weight_sum = classification_loss_sum(output.logits, batch["labels"], class_weights)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}, batch {step}")
            loss.backward()
            epoch_loss += loss.item()
            epoch_weight += weight_sum
            window_weight += weight_sum
            if step % args.accumulation == 0 or step == len(loaders["train"]):
                normalize_gradients(model.parameters(), window_weight)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                window_weight = 0.0
                global_step += 1
                if global_step % 10 == 0 or global_step == 1:
                    elapsed = time.monotonic() - started
                    status = {"state": "training", "epoch": epoch, "step": global_step, "total_steps": total_updates,
                        "train_loss": epoch_loss / epoch_weight, "elapsed_seconds": elapsed,
                        "estimated_training_seconds_remaining": elapsed / global_step * (total_updates - global_step)}
                    write_json(out / "status.json", status)
                    print(json.dumps(status), flush=True)
        validation, _, _ = evaluate(model, loaders["validation"], device, use_bf16)
        row = {"epoch": epoch, "train_loss": epoch_loss / epoch_weight, "validation": validation}
        history.append(row)
        write_json(out / "history.json", history)
        print(f"Epoch {epoch}: validation accuracy={validation['accuracy']:.4f}, macro_f1={validation['macro_f1']:.4f}", flush=True)
        if validation["macro_f1"] > best_score:
            best_score, best_epoch = validation["macro_f1"], epoch
            model.save_pretrained(out / "best_model")
            tokenizer.save_pretrained(out / "best_model")
            write_json(out / "best_model" / "inference_config.json", {"max_length": args.max_length, "rating": "label_id + 1", "best_epoch": best_epoch})
        # A resumable optimizer checkpoint for recovery and inspection.
        checkpoint_tmp = out / "last_checkpoint.tmp"
        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "args": vars(args), "class_weighting": weighting,
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}, checkpoint_tmp)
        checkpoint_tmp.replace(out / "last_checkpoint.pt")
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    best_model = load_model(out / "best_model").to(device)
    test_result, predictions, probabilities = evaluate(best_model, loaders["test"], device, use_bf16)
    majority = int(frames["train"].star_rating.mode().iloc[0])
    result = {"smoke_test_only": args.smoke, "class_weighting": weighting,
              "best_epoch": best_epoch, "best_validation_macro_f1": best_score,
              "test": test_result, "majority_baseline": metrics(frames["test"].star_rating, [majority] * len(frames["test"])),
              "elapsed_seconds": time.monotonic() - started}
    write_json(out / "metrics.json", result)
    pd.DataFrame({"source_row_index": frames["test"].index, "true_stars": frames["test"].star_rating.to_numpy() + 1,
                  "predicted_stars": np.array(predictions) + 1,
                  **{f"p_{i+1}_stars": np.array(probabilities)[:, i] for i in range(5)}}).to_csv(out / "test_predictions.csv", index=False)
    write_json(out / "status.json", {"state": "completed", **result})
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/wongnai")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-weight", choices=["balanced", "none"], default="balanced",
                        help="Inverse class frequency from training labels only (default: balanced)")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.accumulation, args.max_length) < 1 or args.max_length > 512:
        parser.error("Counts must be positive and max-length must be <=512")
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run_config.json").exists():
        parser.error("Output already contains a run; choose a new --output to preserve it")
    write_json(output / "status.json", {"state": "preparing"})
    try:
        run(args, output)
    except BaseException as error:
        write_json(output / "status.json", {"state": "failed", "error": repr(error), "traceback": traceback.format_exc()})
        raise
