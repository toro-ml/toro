# HubSmolLm2

Load the pretrained [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) model from Hugging Face Hub and generate an instruction response on the CPU.

The example pins the model to commit `12fd25f77366fa6b3b4b768ec3050bf629380bac`.

## Run

```bash
nix develop -c dotnet run --project examples/HubSmolLm2
```

Pass a prompt and maximum number of generated tokens:

```bash
nix develop -c dotnet run --project examples/HubSmolLm2 -- "What is 84 * 3 / 2?" 16
```

Example output:

```text
Loading HuggingFaceTB/SmolLM2-135M-Instruct at 12fd25f77366fa6b3b4b768ec3050bf629380bac ...
Loaded 272 tensors.
Prompt: What is 84 * 3 / 2?

To solve this, we first need to simplify the expression inside the parentheses.
```

The first run downloads approximately 269 MB of BF16 weights plus the tokenizer vocabulary and merge rules.
Files are cached by repository and revision under `~/.cache/toro/hub/`.

## Reference regression

The Python reference environment runs the same pinned model with Transformers:

```bash
cd reference_py
uv sync --dev --extra cpu
uv run python llm_reference.py smollm2 "What is 84 * 3 / 2?" 16
```

Both implementations prefill once and then decode one token at a time with a key/value cache.
The Python reference selects Transformers' eager attention so grouped-query attention uses the same explicit K/V expansion as TorchSharp.
Its generated IDs are `[2068, 5482, 451, 28, 392, 808, 737, 288, 21000, 260, 4352, 2972, 260, 38612, 30, 1116]`, which decode to the same response shown above.

## Implementation

- `Toro.Models.SmolLm2` provides the Llama architecture, rotary position embedding, grouped-query attention, SwiGLU MLP, and fixed-capacity KV cache.
- The common generation session performs prefill once and owns incremental decoding state.
- `Toro.Extensions.AI` adapts the model and chat template to `Microsoft.Extensions.AI.IChatClient` and streams response updates.
- The model descriptor uses Hugging Face weight names directly, independently of the F# record layout.
- `SmolLm2.loadFromDirectory` validates the config and complete BF16 state, then copies one tensor at a time.
- `BpeConfig.ByteLevel` and `ByteLevelPreTokenizer` configure the model's GPT-2 tokenizer and chat tokens.
- The example handles only revision-pinned downloads, tokenization, the chat template, greedy decoding, and output.

The example is intended to demonstrate integration on a CPU. It does not provide an optimized LLM inference runtime.
