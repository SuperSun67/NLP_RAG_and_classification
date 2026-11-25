"""Download pinned public model/data assets into this workspace."""
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_REVISION = "7450fab5b23c605484784d1b6fc169461649644b"


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def download(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size:
        print(f"Already present: {target.name}", flush=True)
        return
    temporary = target.with_suffix(target.suffix + ".part")
    print(f"Downloading {target.name}", flush=True)
    with urllib.request.urlopen(url, timeout=180) as response, temporary.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    temporary.replace(target)


def main():
    model_files = get_json("https://huggingface.co/api/models/KordAI/III-BERT/revision/" + MODEL_REVISION)
    for item in model_files["siblings"]:
        name = item["rfilename"]
        if name.endswith((".py", ".json", ".model", ".vocab", ".safetensors", ".md")):
            download(f"https://huggingface.co/KordAI/III-BERT/resolve/{MODEL_REVISION}/{name}", ROOT / "assets" / "iii-bert" / name)
    dataset = get_json("https://huggingface.co/api/datasets/Highgroundbkk/wongnai_reviews")
    revision = dataset["sha"]
    names = [x["rfilename"] for x in dataset["siblings"] if x["rfilename"].endswith(".parquet")]
    if not names:
        raise RuntimeError("No parquet files found")
    for name in names:
        download(f"https://huggingface.co/datasets/Highgroundbkk/wongnai_reviews/resolve/{revision}/{name}", ROOT / "assets" / "wongnai" / name)
    manifest = {"model_id": "KordAI/III-BERT", "model_revision": MODEL_REVISION,
                "dataset_id": "Highgroundbkk/wongnai_reviews", "dataset_revision": revision,
                "dataset_files": names}
    (ROOT / "assets" / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
