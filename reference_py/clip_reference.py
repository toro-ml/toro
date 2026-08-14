import argparse
import io
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, TypedDict

import torch
from PIL import Image
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.processing_auto import AutoProcessor

REPO = "openai/clip-vit-base-patch32"
REVISION = "c7244be81152024ce0e99ac8d2e373a8953d9f9a"
DEFAULT_SOURCE = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/parrots.png"
DEFAULT_LABELS = ["parrot", "bird", "dog"]


class Prediction(TypedDict):
    label: str
    probability: float


class ReferenceResult(TypedDict):
    source: str
    predictions: list[Prediction]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pinned CLIP zero-shot classification as a Toro reference.")
    parser.add_argument("source", nargs="?", default=DEFAULT_SOURCE)
    parser.add_argument("labels", nargs="*", default=DEFAULT_LABELS)
    args = parser.parse_args()

    if not args.labels:
        parser.error("at least one label is required")

    return args


def load_image(source: str) -> Image.Image:
    uri = urllib.parse.urlparse(source)

    if uri.scheme in {"http", "https"}:
        with urllib.request.urlopen(source) as response:
            return Image.open(io.BytesIO(response.read())).convert("RGB")

    return Image.open(Path(source)).convert("RGB")


def run_reference(source: str, labels: list[str]) -> ReferenceResult:
    processor: Any = AutoProcessor.from_pretrained(REPO, revision=REVISION)
    model: Any = AutoModel.from_pretrained(REPO, revision=REVISION, use_safetensors=True)
    prompts = [f"a photo of a {label}" for label in labels]
    inputs = processor(text=prompts, images=load_image(source), return_tensors="pt", padding=True)

    with torch.inference_mode():
        logits = model(**inputs).logits_per_image[0]
        probabilities: list[float] = logits.softmax(dim=-1).tolist()

    predictions: list[Prediction] = [
        {"label": label, "probability": probability} for label, probability in zip(labels, probabilities, strict=True)
    ]
    predictions.sort(key=lambda prediction: prediction["probability"], reverse=True)
    return {"source": source, "predictions": predictions}


def main() -> None:
    args = parse_args()
    result = run_reference(args.source, args.labels)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
