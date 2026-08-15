# Toro

[![CI](https://github.com/toro-ml/toro/actions/workflows/ci.yml/badge.svg)](https://github.com/toro-ml/toro/actions/workflows/ci.yml)
[![Toro](https://img.shields.io/nuget/v/Toro.svg?label=Toro)](https://www.nuget.org/packages/Toro)
[![Toro.NN](https://img.shields.io/nuget/v/Toro.NN.svg?label=Toro.NN)](https://www.nuget.org/packages/Toro.NN)
[![Toro.GNN](https://img.shields.io/nuget/v/Toro.GNN.svg?label=Toro.GNN)](https://www.nuget.org/packages/Toro.GNN)
[![Toro.Text](https://img.shields.io/nuget/v/Toro.Text.svg?label=Toro.Text)](https://www.nuget.org/packages/Toro.Text)
[![Toro.Vision](https://img.shields.io/nuget/v/Toro.Vision.svg?label=Toro.Vision)](https://www.nuget.org/packages/Toro.Vision)

Toro is a machine learning library for F# built on [TorchSharp](https://github.com/dotnet/TorchSharp).
Models can be defined as F# records, and Toro uses TorchSharp tensors directly.
Model structure determines stable names for parameters and buffers, which are also used for optimizer state and checkpoints.

[Documentation](https://toro-ml.github.io/toro/) · [Examples](#examples) · [NuGet packages](#packages)

> [!NOTE]
> Toro is under active development.
> Public APIs and checkpoint formats may change between releases.

## Install

Toro targets .NET 10.
Install the core tensor and neural-network packages with a TorchSharp runtime:

```bash
dotnet add package Toro
dotnet add package Toro.NN
dotnet add package TorchSharp-cpu
```

Add `Toro.GNN`, `Toro.Models`, `Toro.Text`, or `Toro.Vision` when the application needs those features.

## First model

This example defines an F# record model and trains it on XOR.
The `scoped` computation expression disposes intermediate tensors at the end of each iteration.

```fsharp
open TorchSharp
open Toro
open Toro.NN

type Classifier = {
    Fc1: Linear
    Drop: Dropout
    Fc2: Linear
} with

    member this.forward(train: bool) : Tensor -> Tensor =
        _.flatten(1L, -1L)
        >> this.Fc1.forward
        >> _.relu()
        >> this.Drop.forwardT train
        >> this.Fc2.forward

let x =
    torch.tensor (
        array2D [|
            [| 0f; 0f |]
            [| 0f; 1f |]
            [| 1f; 0f |]
            [| 1f; 1f |]
        |],
        device = torch.CPU
    )

let y =
    torch.tensor (
        array2D [| [| 0f |]; [| 1f |]; [| 1f |]; [| 0f |] |],
        device = torch.CPU
    )

let model = {
    Fc1 = Linear.init 2 16 torch.float32 torch.CPU
    Drop = Dropout.create 0.1
    Fc2 = Linear.init 16 1 torch.float32 torch.CPU
}

let modelState = Model.state model
let optimizer = AdamW.createWithLr 0.01 (ModelState.trainableParams modelState)

for epoch in 1..500 do
    scoped {
        optimizer.zeroGrad ()
        let prediction = model.forward true x
        let loss = Loss.mse prediction y
        loss.backward ()
        optimizer.step ()

        if epoch % 100 = 0 then
            printfn "epoch %d  loss=%.6f" epoch (loss.ToSingle())
    }
```

## Model state

Toro discovers model state recursively through records, options, tuples, discriminated unions, arrays, F# lists, `ResizeArray`, `IReadOnlyList`, and string-keyed dictionaries.
Tensor fields must declare whether they are trainable parameters, persistent buffers, or ignored values.
Built-in layers already contain these annotations.

```fsharp
type NormalizedScale = {
    [<Parameter>]
    Scale: Tensor

    [<Buffer>]
    RunningMean: Tensor

    [<ModelIgnore>]
    Scratch: Tensor
}
```

`Model.state` creates a validated state view.
`ModelState.namedState` returns canonical names for parameters and buffers, while `ModelState.trainableParams` returns the gradient-enabled parameters accepted by `SGD` and `AdamW`.
Shared tensors are registered once, preventing duplicate optimizer updates and duplicate checkpoint entries.

## External weights

`NameMapping` describes how external tensor names map onto an F# model.
Rules can rename exact paths, rewrite complete path segments with captures, or ignore a known suffix.

```fsharp
let mapping =
    NameMapping.create [
        NameRule.rewrite
            "encoder.layer.{layer}.weight"
            "Layers.{layer}.Weight"

        NameRule.ignoreSuffix "num_batches_tracked"
    ]

let report =
    weights
    |> ModelState.loadFromDictWith mapping Strict (Model.state model)
```

Name ambiguity, target collisions, missing keys, unexpected keys, shapes, and dtypes are validated before tensors are copied.
The [HubResNet18](examples/HubResNet18) and [HubSentiment](examples/HubSentiment) examples load weights from pinned Hugging Face revisions.

## Training state

`Checkpoint.save` and `Checkpoint.load` store canonical model state, optimizer state, epoch, learning rate, and optimizer kind.
AdamW state uses parameter names rather than parameter positions.

Reproducible training also requires the random-number-generator and scheduler states owned by the training loop.
The [MnistTraining](examples/MnistTraining) example saves CPU Torch RNG state and `SchedulerState`, recreates each shuffled DataLoader from an epoch seed, and resumes at epoch boundaries.

## Features

- **TorchSharp tensors:** Toro exposes `torch.Tensor` directly and adds typed indexing, comparison operators, and lifetime helpers.
- **Scoped ownership:** `scoped { }` disposes intermediate tensors while preserving tensors returned in records, tuples, lists, options, and unions.
- **F# model composition:** Define models with records, `IModule`, `sequential { }`, and `pipeline { }`.
- **Neural networks:** Linear, convolution, normalization, recurrent, attention, pooling, activation, and loss modules.
- **Named optimization:** SGD and AdamW validate canonical parameter names and persist optimizer state without positional coupling.
- **SafeTensors:** Save canonical parameters and buffers, or load external weights with strict preflight validation.
- **Vision and text:** Image loading and transforms plus tokenization based on Microsoft.ML.Tokenizers.
- **Graph neural networks:** Message passing, GCN, GAT, GraphSAGE, GIN, graph normalization, and global pooling.

## Packages

| Package | Purpose |
| --- | --- |
| [`Toro`](src/Toro) | Tensor extensions, scoped ownership, and SafeTensors |
| [`Toro.NN`](src/Toro.NN) | Model state, layers, optimizers, schedulers, and checkpoints |
| [`Toro.GNN`](src/Toro.GNN) | Graph data, message passing, graph convolutions, and pooling |
| [`Toro.Models`](src/Toro.Models) | Pretrained models, local loaders, and token-level generation |
| [`Toro.Text`](src/Toro.Text) | Tokenization and tensor encoding |
| [`Toro.Vision`](src/Toro.Vision) | Image loading and tensor transforms |

`Toro.Hub` downloads one revision-pinned Hugging Face file at a time and caches it locally.

## Examples

| Example | Demonstrates |
| --- | --- |
| [LinearRegression](examples/LinearRegression) | Gradient descent with tensors |
| [SimpleTraining](examples/SimpleTraining) | XOR training with `sequential { }` |
| [MnistTraining](examples/MnistTraining) | Reproducible CNN training and checkpoint resume |
| [MnistCnn](examples/MnistCnn) | CNN composition with BatchNorm and Dropout |
| [MnistAutoencoder](examples/MnistAutoencoder) | Autoencoder training and image output |
| [MnistGan](examples/MnistGan) | Adversarial training with independent optimizers |
| [CharRnn](examples/CharRnn) | Character-level generation with LSTM |
| [TextClassifier](examples/TextClassifier) | Transformer-based text classification |
| [SimpleGcn](examples/SimpleGcn) | Node classification with GCNConv |
| [HubSentiment](examples/HubSentiment) | Pinned DistilBERT weights and declarative name mapping |
| [HubResNet18](examples/HubResNet18) | Pinned ResNet-18 weights and image preprocessing |
| [HubClip](examples/HubClip) | Zero-shot image classification with pinned CLIP weights |
| [HubDistilGpt2](examples/HubDistilGpt2) | CPU text generation with pinned DistilGPT2 weights |
| [HubSmolLm2](examples/HubSmolLm2) | CPU instruction generation with pinned SmolLM2 weights |

## Development

Use the repository's Nix development environment:

```bash
nix develop -c dotnet tool restore
nix develop -c fantomas src tests examples scripts
nix develop -c dotnet build Toro.slnx
nix develop -c dotnet test Toro.slnx
```

Preview the documentation site locally:

```bash
cd docs
pnpm install
pnpm dev
```

## License

[MIT](LICENSE)
