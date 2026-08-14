import argparse
import json
from dataclasses import dataclass
from typing import Any, TypedDict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class ModelSpec:
    repo: str
    revision: str
    dtype: torch.dtype
    chat: bool


class ReferenceResult(TypedDict):
    input_ids: list[int]
    response_ids: list[int]
    response: str


MODELS = {
    "distilgpt2": ModelSpec(
        repo="distilbert/distilgpt2",
        revision="2290a62682d06624634c1f46a6ad5be0f47f38aa",
        dtype=torch.float32,
        chat=False,
    ),
    "smollm2": ModelSpec(
        repo="HuggingFaceTB/SmolLM2-135M-Instruct",
        revision="12fd25f77366fa6b3b4b768ec3050bf629380bac",
        dtype=torch.bfloat16,
        chat=True,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a pinned Transformers model as a Toro reference.")
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("prompt")
    parser.add_argument("max_new_tokens", type=int)
    args = parser.parse_args()

    if args.max_new_tokens < 0:
        parser.error("max_new_tokens must be non-negative")

    return args


def encode_prompt(tokenizer: Any, spec: ModelSpec, prompt: str) -> Any:
    if not spec.chat:
        return tokenizer(prompt, return_tensors="pt")

    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    return tokenizer(text, return_tensors="pt", add_special_tokens=False)


def run_reference(spec: ModelSpec, prompt: str, max_new_tokens: int) -> ReferenceResult:
    tokenizer: Any = AutoTokenizer.from_pretrained(spec.repo, revision=spec.revision)
    model: Any = AutoModelForCausalLM.from_pretrained(
        spec.repo,
        revision=spec.revision,
        dtype=spec.dtype,
        use_safetensors=True,
    )
    inputs = encode_prompt(tokenizer, spec, prompt)

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=False,
    )
    response = output[0, inputs.input_ids.shape[1] :]
    return {
        "input_ids": inputs.input_ids[0].tolist(),
        "response_ids": response.tolist(),
        "response": tokenizer.decode(response, skip_special_tokens=True),
    }


def main() -> None:
    args = parse_args()
    result = run_reference(MODELS[args.model], args.prompt, args.max_new_tokens)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
