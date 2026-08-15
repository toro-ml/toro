# HubResNet18

Load the pretrained [Microsoft ResNet-18](https://huggingface.co/microsoft/resnet-18) from Hugging Face Hub and classify an image on the CPU.

The example pins the model to commit `65a5785d9156231087c481e0c7dd33a5ff6f7e3e` so that the weights, preprocessing configuration, and labels are reproducible.

## Run

Classify the default parrot image:

```bash
nix develop -c dotnet run --project examples/HubResNet18
```

Pass a local image path or an HTTP(S) URL as the first argument:

```bash
nix develop -c dotnet run --project examples/HubResNet18 -- ./image.jpg
```

The first run downloads `model.safetensors`, `preprocessor_config.json`, and `config.json`.
Files are cached by repository and revision under `~/.cache/toro/hub/`.

## Example output

```text
Loading microsoft/resnet-18 at 65a5785d9156231087c481e0c7dd33a5ff6f7e3e ...
Loaded 102 parameters and buffers; ignored 20 source tensors.
Predictions for https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/parrots.png:
  1. macaw                          62.96%
  2. partridge                      6.63%
  3. ruffed grouse, partridge, Bonasa umbellus 5.50%
  4. vulture                        5.30%
  5. quail                          4.64%
```

Exact probabilities can vary with the TorchSharp and LibTorch versions.

## How it works

1. `Hub.download` downloads the weights at the pinned revision.
2. F# records define the ResNet-18 stem, residual blocks, stages, and classifier.
3. `NameMapping` rewrites Hugging Face stage, block, and layer paths into Toro model paths.
4. `NameRule.ignoreSuffix "num_batches_tracked"` explicitly excludes the PyTorch BatchNorm counters that Toro does not store.
5. `ModelState.loadSafeTensorsWith` uses `Strict` mode to validate all remaining parameter and buffer names, shapes, and dtypes before copying tensors one at a time.
6. The pinned preprocessing configuration supplies resize size, crop ratio, channel means, and channel standard deviations.
7. `ResizeShortestEdge`, center crop, ImageNet normalization, and softmax produce the top five labels.

The model-specific record types, name mapping, and JSON handling remain in this example rather than becoming a generic model registry.
