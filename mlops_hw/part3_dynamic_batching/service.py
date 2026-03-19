from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer

# Configuration

MODEL_NAME = "cointegrated/rubert-tiny2"
ONNX_PATH = Path(__file__).parent.parent / "models" / "rubert_mini.onnx"
MAX_WAIT_MS: float = 20.0    # collect for up to 20 ms
MAX_BATCH_SIZE: int = 32     # or until this many texts are queued

# Model loading

print(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print(f"Loading ONNX model: {ONNX_PATH}")
_sess_opts = ort.SessionOptions()
_sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession(
    str(ONNX_PATH),
    sess_options=_sess_opts,
    providers=["CPUExecutionProvider"],
)
_input_names: set[str] = {inp.name for inp in session.get_inputs()}
print(f"ONNX model ready. Inputs: {_input_names}")

# Shared async state (initialised at startup)

_queue: asyncio.Queue[tuple[list[str], asyncio.Future[list[list[float]]]]] | None = None
_worker_task: asyncio.Task | None = None


# Helpers
def _mean_pool(embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    mask = np.expand_dims(attention_mask, -1).astype(np.float32)
    return np.sum(embeddings * mask, axis=1) / np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)


def _run_onnx(texts: list[str]) -> np.ndarray:
    """Synchronous ONNX inference.  Returns shape (N, hidden_dim)."""
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
    return embs / np.clip(norms, a_min=1e-9, a_max=None)


# Background batching worker

async def _batching_worker() -> None:
    """Drain the queue, accumulate texts, run one ONNX call, fan results back."""
    loop = asyncio.get_running_loop()
    while True:
        # Wait for the first request
        try:
            first_texts, first_future = await _queue.get()  # type: ignore[union-attr]
        except asyncio.CancelledError:
            return

        # Accumulate more requests until timeout or batch full
        batch_texts: list[str] = list(first_texts)
        pending: list[tuple[asyncio.Future[list[list[float]]], int]] = [
            (first_future, len(first_texts))
        ]

        deadline = loop.time() + MAX_WAIT_MS / 1000.0

        while len(batch_texts) < MAX_BATCH_SIZE:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                texts, future = await asyncio.wait_for(
                    _queue.get(),  # type: ignore[union-attr]
                    timeout=remaining,
                )
            except asyncio.TimeoutError:
                break

            if len(batch_texts) + len(texts) > MAX_BATCH_SIZE:
                # Re-queue and stop accumulating.
                await _queue.put((texts, future))  # type: ignore[union-attr]
                break

            batch_texts.extend(texts)
            pending.append((future, len(texts)))

        # Run batched inference in a thread pool (keeps event loop free)
        try:
            embeddings: np.ndarray = await loop.run_in_executor(
                None, _run_onnx, batch_texts
            )
        except Exception as exc:
            for fut, _ in pending:
                if not fut.done():
                    fut.set_exception(exc)
            continue

        # Fan results back to each waiting coroutine
        idx = 0
        for fut, count in pending:
            if not fut.done():
                fut.set_result(embeddings[idx : idx + count].tolist())
            idx += count


# Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _queue, _worker_task
    _queue = asyncio.Queue()
    _worker_task = asyncio.create_task(_batching_worker())
    yield
    _worker_task.cancel()
    try:
        await _worker_task
    except asyncio.CancelledError:
        pass


# FastAPI app
app = FastAPI(title="Part 3 — Dynamic-Batching ONNX Embedding Service", lifespan=lifespan)


class EmbedRequest(BaseModel):
    texts: list[str]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/embed")
async def embed(req: EmbedRequest) -> dict:
    if _queue is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    loop = asyncio.get_running_loop()
    future: asyncio.Future[list[list[float]]] = loop.create_future()
    await _queue.put((req.texts, future))
    embeddings = await future
    return {"embeddings": embeddings}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
