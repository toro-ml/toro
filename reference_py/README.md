
# Python reference environment

This environment contains Python reference implementations used to check Toro examples against upstream libraries.

## Setup

```
uv sync --dev --extra cpu
```

## LLM references

Run the pinned Transformers model with greedy decoding and print input IDs, generated IDs, and decoded text as JSON:

```bash
uv run python llm_reference.py distilgpt2 "Hello, I'm a language model" 8
uv run python llm_reference.py smollm2 "What is 84 * 3 / 2?" 16
```

The reference disables the Transformers KV cache because the Toro examples recompute the full sequence for each token.
This keeps the BF16 operation order comparable for SmolLM2.
