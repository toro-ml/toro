# Toro.NN

[![Toro.NN](https://img.shields.io/nuget/v/Toro.NN.svg?label=Toro.NN)](https://www.nuget.org/packages/Toro.NN)

Neural network building blocks for [Toro](https://www.nuget.org/packages/Toro). Layers, composition, optimizers, and training utilities with F# records and `result { }` error handling.

**[Documentation](https://toro-ml.github.io/toro/)**

## Installation

```bash
dotnet add package Toro.NN
dotnet add package TorchSharp-cpu
```

## Quick Example

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
}
```

## Features

- **Layers** -- Linear, Conv1d/Conv2d, Embedding, BatchNorm, LayerNorm, GroupNorm, Dropout, pooling, activations
- **Blocks** -- LSTM, GRU, MultiHeadAttention, TransformerBlock, KV cache
- **Composition** -- `sequential { }`, `pipeline { }`, and `>=>` Kleisli composition
- **Training** -- SGD, AdamW, learning-rate schedulers, loss functions, metrics, gradient clipping, checkpoints

## License

[MIT](https://github.com/toro-ml/toro/blob/main/LICENSE)
