---
title: Toro
category: Documentation
categoryindex: 1
index: 0
---

# Toro

Toro is a minimalist ML framework for F#.
It uses TorchSharp as its tensor backend.
You define models with F# records and computation expressions.

## Installation

```bash
dotnet add package Toro
dotnet add package Toro.NN
```

Add a TorchSharp runtime package for your platform:

```bash
dotnet add package TorchSharp-cpu
```

## Minimal Example

Train a linear model on the XOR problem:

```fsharp
open Toro
open Toro.NN

let r = result {
    let! l1 = Linear.init 2 16 F32 Cpu
    let! l2 = Linear.init 16 1 F32 Cpu
    let model = sequential { l1; Relu; l2 }

    let! opt = AdamW.createWithLr 0.01 (Model.trainableVars model)
    let opt = opt :> IOptimizer

    for epoch in 1..500 do
        let! pred = model.forward x
        let! loss = Loss.mse pred y
        do! opt.backwardStep loss
}
```

The `result { }` computation expression chains operations that return `Result<'T, ToroError>`.
Use `let!` to unwrap each result. Use `do!` for operations that return `Result<unit, ToroError>`.

## Documentation

- [Getting Started](getting-started.html) -- Install Toro and train your first model
- [Tensor](tensor.html) -- Create tensors, do arithmetic, reshape, and index
- [Neural Networks](nn.html) -- Layers, modules, Sequential, and SequentialT
- [Training](training.html) -- Loss functions, optimizers, and the training loop

## Examples

The repository has these examples:

| Example | Description |
| --- | --- |
| [LinearRegression](https://github.com/toro-ml/toro/tree/main/examples/LinearRegression) | Gradient descent with raw tensors |
| [SimpleTraining](https://github.com/toro-ml/toro/tree/main/examples/SimpleTraining) | XOR with `sequential { }` CE |
| [MnistTraining](https://github.com/toro-ml/toro/tree/main/examples/MnistTraining) | CNN image classification |
| [MnistCnn](https://github.com/toro-ml/toro/tree/main/examples/MnistCnn) | CNN with BatchNorm, Dropout, `sequentialT { }` |
| [MnistAutoencoder](https://github.com/toro-ml/toro/tree/main/examples/MnistAutoencoder) | Autoencoder with image output |
| [MnistGan](https://github.com/toro-ml/toro/tree/main/examples/MnistGan) | GAN image generation |
| [CharRnn](https://github.com/toro-ml/toro/tree/main/examples/CharRnn) | Character-level text generation with LSTM |
| [TextClassifier](https://github.com/toro-ml/toro/tree/main/examples/TextClassifier) | Transformer-based text classification |
