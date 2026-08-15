import argparse
import json
from dataclasses import dataclass
from typing import TypedDict, cast

import torch
from transformers.cache_utils import Cache
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase


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


def encode_prompt(tokenizer: PreTrainedTokenizerBase, spec: ModelSpec, prompt: str) -> BatchEncoding:
    if not spec.chat:
        return tokenizer(prompt, return_tensors="pt")

    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )

    if not isinstance(text, str):
        raise TypeError("The chat template must produce text when tokenize=False.")

    return tokenizer(text, return_tensors="pt", add_special_tokens=False)


def greedy_generate(
    model: GPT2LMHeadModel | LlamaForCausalLM,
    input_ids: torch.Tensor,
    eos_token_id: int,
    max_new_tokens: int,
) -> torch.Tensor:
    generated = input_ids
    current_input = input_ids
    past_key_values: Cache | None = None

    for _ in range(max_new_tokens):
        output = model.forward(input_ids=current_input, past_key_values=past_key_values, use_cache=True)
        next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat((generated, next_token), dim=1)

        if output.past_key_values is None:
            raise RuntimeError("The model did not return a key/value cache.")

        past_key_values = output.past_key_values
        current_input = next_token

        if next_token.item() == eos_token_id:
            break

    return generated


def run_reference(spec: ModelSpec, prompt: str, max_new_tokens: int) -> ReferenceResult:
    tokenizer = AutoTokenizer.from_pretrained(spec.repo, revision=spec.revision)

    if tokenizer is None:
        raise RuntimeError(f"No tokenizer is available for {spec.repo}.")

    model_type = LlamaForCausalLM if spec.chat else GPT2LMHeadModel

    model = model_type.from_pretrained(
        spec.repo,
        revision=spec.revision,
        dtype=spec.dtype,
        use_safetensors=True,
        attn_implementation="eager",
    )
    inputs = encode_prompt(tokenizer, spec, prompt)
    input_ids = inputs["input_ids"]

    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("The tokenizer must return input_ids as a tensor.")

    eos_token_id = tokenizer.eos_token_id

    if not isinstance(eos_token_id, int):
        raise TypeError("The tokenizer must define one EOS token ID.")

    output = greedy_generate(model, input_ids, eos_token_id, max_new_tokens)
    response = output[0, input_ids.shape[1] :]
    input_id_values = cast(list[int], input_ids[0].tolist())
    response_id_values = cast(list[int], response.tolist())
    decoded = tokenizer.decode(response_id_values, skip_special_tokens=True)

    if not isinstance(decoded, str):
        raise TypeError("Decoding one token sequence must produce text.")

    return {
        "input_ids": input_id_values,
        "response_ids": response_id_values,
        "response": decoded,
    }


def main() -> None:
    args = parse_args()
    result = run_reference(MODELS[args.model], args.prompt, args.max_new_tokens)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
