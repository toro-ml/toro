# HubClip

Load the pretrained [CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32) model from Hugging Face Hub and classify an image against user-supplied labels on the CPU.

The example pins the model to commit `c7244be81152024ce0e99ac8d2e373a8953d9f9a`.

## Run

Classify the default parrot image against `parrot`, `bird`, and `dog`:

```bash
nix develop -c dotnet run --project examples/HubClip
```

Pass a local image path or HTTP(S) URL followed by candidate labels:

```bash
nix develop -c dotnet run --project examples/HubClip -- ./image.jpg "tabby cat" dog car
```

Each label is inserted into the prompt `a photo of a {label}`.
The first run downloads approximately 605 MB of F32 weights plus the CLIP vocabulary and merge rules.
Files are cached by repository and revision under `~/.cache/toro/hub/`.

Example output:

```text
Loading openai/clip-vit-base-patch32 at c7244be81152024ce0e99ac8d2e373a8953d9f9a ...
Loaded 398 tensors.
Zero-shot predictions for https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/parrots.png:
  1. parrot                   72.02%
  2. bird                     27.95%
  3. dog                      0.03%
```

## Reference regression

The Python reference environment runs the same pinned model and preprocessing with Transformers:

```bash
cd reference_py
uv sync --dev --extra cpu
uv run python clip_reference.py
```

Use the same image and labels for both implementations:

```bash
# From the repository root
nix develop -c dotnet run --project examples/HubClip -- ./image.jpg "tabby cat" dog car

cd reference_py
uv run python clip_reference.py ../image.jpg "tabby cat" dog car
```

The default input produces the following comparison:

| Rank | Label | Toro | Transformers |
| ---: | --- | ---: | ---: |
| 1 | `parrot` | 72.02% | 72.14% |
| 2 | `bird` | 27.95% | 27.83% |
| 3 | `dog` | 0.03% | 0.03% |

The ranked labels match.
Small probability differences can result from the different image resize implementations and from TorchSharp/LibTorch versus Transformers/PyTorch operations.

## Implementation

- F# records define the text Transformer, vision Transformer, and projection layers inside the example.
- The text side uses the pinned CLIP byte-level BPE vocabulary and merge rules.
- `NameMapping` maps Hugging Face tensor paths to the F# record paths before strict loading.
- The two serialized `position_ids` tensors are ignored explicitly because both encoders generate them at runtime.
- Image preprocessing uses a bicubic shortest-edge resize, center crop, and the CLIP channel normalization constants.
- Text and image embeddings are L2-normalized before applying CLIP's learned logit scale.

The CLIP architecture and prompt construction remain in this example rather than becoming a model registry or generic zero-shot API.
