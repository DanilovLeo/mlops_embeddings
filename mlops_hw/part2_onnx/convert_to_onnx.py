from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "cointegrated/rubert-tiny2"
MODELS_DIR = Path(__file__).parent.parent / "models"
ONNX_PATH = MODELS_DIR / "rubert_mini.onnx"


class _ModelWrapper(torch.nn.Module):

    def __init__(self, model: AutoModel, has_token_type_ids: bool) -> None:
        super().__init__()
        self.model = model
        self.has_token_type_ids = has_token_type_ids

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kwargs: dict = {"input_ids": input_ids, "attention_mask": attention_mask}
        if self.has_token_type_ids and token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        return self.model(**kwargs).last_hidden_state


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModel.from_pretrained(MODEL_NAME)
    base_model.eval()

    sample = ["Это тестовый текст.", "Hello world"]
    enc = tokenizer(sample, return_tensors="pt", padding=True, truncation=True, max_length=128)

    has_token_type_ids = "token_type_ids" in enc
    wrapped = _ModelWrapper(base_model, has_token_type_ids)
    wrapped.eval()

    if has_token_type_ids:
        dummy_inputs = (enc["input_ids"], enc["attention_mask"], enc["token_type_ids"])
        input_names = ["input_ids", "attention_mask", "token_type_ids"]
        dynamic_axes = {
            "input_ids":       {0: "batch_size", 1: "seq_len"},
            "attention_mask":  {0: "batch_size", 1: "seq_len"},
            "token_type_ids":  {0: "batch_size", 1: "seq_len"},
            "last_hidden_state": {0: "batch_size", 1: "seq_len"},
        }
    else:
        dummy_inputs = (enc["input_ids"], enc["attention_mask"])
        input_names = ["input_ids", "attention_mask"]
        dynamic_axes = {
            "input_ids":       {0: "batch_size", 1: "seq_len"},
            "attention_mask":  {0: "batch_size", 1: "seq_len"},
            "last_hidden_state": {0: "batch_size", 1: "seq_len"},
        }

    print(f"Exporting ONNX → {ONNX_PATH}")
    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            dummy_inputs,
            str(ONNX_PATH),
            input_names=input_names,
            output_names=["last_hidden_state"],
            dynamic_axes=dynamic_axes,
            opset_version=14,
            do_constant_folding=True,
        )

    # Verify the exported model.
    import onnxruntime as ort

    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    print("Verification passed.")
    print(f"  Inputs:  {[i.name for i in sess.get_inputs()]}")
    print(f"  Outputs: {[o.name for o in sess.get_outputs()]}")
    print(f"Model saved to {ONNX_PATH}")


if __name__ == "__main__":
    main()
