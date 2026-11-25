"""Local inference UI. Run with the project Python: python -X utf8 app.py."""
import argparse
from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from predict import ROOT, ReviewPredictor

logger = logging.getLogger(__name__)


class ReviewRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000, strict=True)

    @field_validator("text")
    @classmethod
    def require_text(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("กรุณากรอกข้อความรีวิว")
        return value


def create_app(model_path=ROOT / "runs/wongnai/best_model", device="auto"):
    model_path = Path(model_path)

    @asynccontextmanager
    async def lifespan(app):
        app.state.predictor = None
        app.state.error = None
        try:
            app.state.predictor = await run_in_threadpool(ReviewPredictor, model_path, device)
        except Exception:
            logger.exception("Could not load the inference checkpoint")
            app.state.error = "ยังโหลดโมเดลไม่ได้ กรุณาตรวจ checkpoint แล้วเริ่มเซิร์ฟเวอร์ใหม่"
        yield
        app.state.predictor = None

    app = FastAPI(title="Wongnai Review Studio", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=ROOT / "ui"), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(ROOT / "ui/index.html")

    @app.get("/api/status")
    def status():
        predictor = app.state.predictor
        evaluation = None
        metrics_path = model_path.parent / "metrics.json"
        if metrics_path.exists():
            try:
                data = json.loads(metrics_path.read_text(encoding="utf-8"))
                if not data.get("smoke_test_only", True):
                    evaluation = {"accuracy": data["test"]["accuracy"], "macro_f1": data["test"]["macro_f1"]}
            except (OSError, ValueError, KeyError, TypeError):
                logger.warning("Evaluation metrics unavailable")
        return {"ready": predictor is not None, "error": app.state.error,
                "model": "III-BERT · Wongnai", "device": str(predictor.device) if predictor else None,
                "max_length": predictor.max_length if predictor else None,
                "best_epoch": predictor.best_epoch if predictor else None,
                "evaluation": evaluation}

    @app.post("/api/predict")
    def predict(review: ReviewRequest):
        if app.state.predictor is None:
            raise HTTPException(status_code=503, detail=app.state.error)
        started = perf_counter()
        try:
            result = app.state.predictor.predict([review.text])[0]
        except Exception:
            logger.exception("Inference failed")
            raise HTTPException(status_code=500, detail="วิเคราะห์ไม่สำเร็จ กรุณาลองใหม่หรือตรวจ log ของเซิร์ฟเวอร์") from None
        return {**result, "elapsed_ms": round((perf_counter() - started) * 1000, 1)}

    return app


if __name__ == "__main__":
    import torch
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="runs/wongnai/best_model")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    torch.set_num_threads(4)
    uvicorn.run(create_app(ROOT / args.model, args.device), host="127.0.0.1", port=args.port, access_log=False)
