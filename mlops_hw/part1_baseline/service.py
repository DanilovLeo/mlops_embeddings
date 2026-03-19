from __future__ import annotations

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "cointegrated/rubert-tiny2"

app = FastAPI(title="Part 1 — Baseline PyTorch Embedding Service")

print(f"Loading model: {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()
print("Model ready.")


class EmbedRequest(BaseModel):
    texts: list[str]


def _mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/embed")
def embed(req: EmbedRequest) -> dict:
    enc = tokenizer(
        req.texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    with torch.no_grad():
        out = model(**enc)
    embs = _mean_pool(out.last_hidden_state, enc["attention_mask"])
    embs = torch.nn.functional.normalize(embs, p=2, dim=1)
    return {"embeddings": embs.tolist()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
