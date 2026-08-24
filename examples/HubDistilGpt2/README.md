# HubDistilGpt2

Load the pretrained [DistilGPT2](https://huggingface.co/distilbert/distilgpt2) model from Hugging Face Hub and generate text on the CPU.

The example pins the model to commit `2290a62682d06624634c1f46a6ad5be0f47f38aa`.

## Run

```bash
nix develop -c dotnet run --project examples/HubDistilGpt2
```

Pass a prompt and maximum number of generated tokens:

```bash
nix develop -c dotnet run --project examples/HubDistilGpt2 -- "Hello, I'm a language model" 8
```

Example output:

```text
Loading distilbert/distilgpt2 at 2290a62682d06624634c1f46a6ad5be0f47f38aa ...
Loaded 76 tensors; ignored 6 checkpoint buffers.
Prompt: Hello, I'm a language model

Hello, I'm a language model. I'm a language model. I
```

The first run downloads approximately 353 MB of F32 weights plus the GPT-2 vocabulary and merge rules.
Files are cached by repository and revision under `~/.cache/toro/hub/`.

## Reference regression

The Python reference environment runs the same pinned model with Transformers:

```bash
cd reference_py
uv sync --dev --extra cpu
uv run python llm_reference.py distilgpt2 "Hello, I'm a language model" 8
```

The reference result has response IDs `[13, 314, 1101, 257, 3303, 2746, 13, 314]` and the suffix `. I'm a language model. I`, matching the Toro output above.
Both implementations prefill once and then decode one token at a time with a key/value cache.

## Implementation

- `Toro.Models.DistilGpt2` provides the GPT-2 architecture and fixed-capacity KV cache; `Toro.Models` provides the common generation session.
- The model descriptor uses Hugging Face names and Conv1D weight shapes directly, so loading does not transpose weights.
- `DistilGpt2.loadFromDirectory` validates the config and complete F32 state, then copies one tensor at a time.
- The stored causal-mask buffers are ignored because TorchSharp creates the causal mask during attention.
- `BpeConfig.ByteLevel` and `ByteLevelPreTokenizer` configure GPT-2 byte-level tokenization.
- The example handles only revision-pinned downloads, tokenization, greedy generation, and output.

DistilGPT2 is an English text-completion model rather than an instruction-following chat model.
