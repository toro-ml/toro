# Toro

[![NuGet](https://img.shields.io/nuget/v/Toro.svg)](https://www.nuget.org/packages/Toro)
[![NuGet](https://img.shields.io/nuget/v/Toro.NN.svg)](https://www.nuget.org/packages/Toro.NN)

PyTorch semantics, idiomatic F#. Powered by  [TorchSharp](https://github.com/dotnet/TorchSharp).

Define models with F# records and computation expressions.
Chain fallible operations with the `result { }` CE.

**[Documentation](https://toro-ml.github.io/toro/)**

## Installation

```bash
dotnet add package Toro
dotnet add package Toro.NN
dotnet add package TorchSharp-cpu
```

## Quick Example

Train a two-layer network on the XOR problem:

```fsharp
open Toro
open Toro.NN

let r = result {
    let! x = Tensor.ofList ([ [ 0f; 0f ]; [ 0f; 1f ]; [ 1f; 0f ]; [ 1f; 1f ] ], Cpu)
    let! y = Tensor.ofList ([ [ 0f ]; [ 1f ]; [ 1f ]; [ 0f ] ], Cpu)

    let! l1 = Linear.init 2 16 F32 Cpu
    let! l2 = Linear.init 16 1 F32 Cpu
    let model = sequential { l1; Relu; l2 }

    let! opt = AdamW.createWithLr 0.01 (Model.trainableVars model)

    for epoch in 1..500 do
        opt.zeroGrad ()
        let! pred = model.forward x
        let! loss = Loss.mse pred y
        do! loss.backward ()
        do! opt.step ()

        if epoch % 100 = 0 then
            printfn "epoch %d  loss=%.6f" epoch (loss.item ())
}
```

## Features

- **Tensor API** -- Create, reshape, index, and compute with tensors. Arithmetic operators return `Tensor` directly. Shape and math methods return `Result<Tensor, ToroError>`.
- **Indexing** -- `t[0]`, `t[0..2]`, and `t.at [ I 1; S(0, 3) ]` for advanced patterns.
- **Neural network layers** -- Linear, Conv1d/Conv2d, Embedding, LSTM, GRU, BatchNorm, Dropout, LayerNorm, MultiHeadAttention, TransformerBlock.
- **Composition** -- `sequential { }` and `sequentialT { }` CEs to build models without casts.
- **Training** -- SGD, AdamW optimizers. MSE, cross-entropy, NLL, binary cross-entropy loss functions.
- **Error handling** -- `result { }` CE with `let!` / `do!` for `Result` chaining.

## Examples

| Example | Description |
| --- | --- |
| [LinearRegression](examples/LinearRegression) | Gradient descent with raw tensors |
| [SimpleTraining](examples/SimpleTraining) | XOR with `sequential { }` CE |
| [MnistTraining](examples/MnistTraining) | MLP on MNIST |
| [MnistCnn](examples/MnistCnn) | CNN with BatchNorm, Dropout, `sequentialT { }` |
| [MnistAutoencoder](examples/MnistAutoencoder) | Autoencoder with image output |
| [MnistGan](examples/MnistGan) | GAN image generation |
| [CharRnn](examples/CharRnn) | Character-level text generation with LSTM |
| [TextClassifier](examples/TextClassifier) | Transformer-based text classification |
| [SimpleGcn](examples/SimpleGcn) | GNN node classification with GCNConv |
| [HubSentiment](examples/HubSentiment) | Load DistilBERT from Hugging Face Hub |

## Development

Build and run tests:

```bash
dotnet build
dotnet test
```

Preview the documentation site locally:

```bash
cd docs
pnpm install
pnpm dev
```

## License

[MIT](LICENSE)
