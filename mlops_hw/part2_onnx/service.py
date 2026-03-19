from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer

MODEL_NAME = "cointegrated/rubert-tiny2"
ONNX_PATH = Path(__file__).parent.parent / "models" / "rubert_mini.onnx"

app = FastAPI(title="Part 2 — ONNX Embedding Service")

print(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print(f"Loading ONNX model: {ONNX_PATH}")
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession(
    str(ONNX_PATH),
    sess_options=sess_options,
    providers=["CPUExecutionProvider"],
)
_input_names: set[str] = {inp.name for inp in session.get_inputs()}
print(f"ONNX model ready. Inputs: {_input_names}")


class EmbedRequest(BaseModel):
    texts: list[str]


def _mean_pool(embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    mask = np.expand_dims(attention_mask, -1).astype(np.float32)
    return np.sum(embeddings * mask, axis=1) / np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)


def _run_onnx(texts: list[str]) -> list[list[float]]:
    enc = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="np")
    feeds: dict[str, np.ndarray] = {
        "input_ids": enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
    }
    if "token_type_ids" in _input_names and "token_type_ids" in enc:
        feeds["token_type_ids"] = enc["token_type_ids"].astype(np.int64)

    (hidden_state,) = session.run(["last_hidden_state"], feeds)
    embs = _mean_pool(hidden_state, enc["attention_mask"])
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / np.clip(norms, a_min=1e-9, a_max=None)
    return embs.tolist()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/embed")
def embed(req: EmbedRequest) -> dict:
    return {"embeddings": _run_onnx(req.texts)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
