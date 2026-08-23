# Toro.NN

[![Toro.NN](https://img.shields.io/nuget/v/Toro.NN.svg?label=Toro.NN)](https://www.nuget.org/packages/Toro.NN)

Neural network building blocks for [Toro](https://www.nuget.org/packages/Toro). Layers, composition, optimizers, and training utilities with F# records.

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

let x = Tensor.ofList ([ [ 0f; 0f ]; [ 0f; 1f ]; [ 1f; 0f ]; [ 1f; 1f ] ], Cpu)
let y = Tensor.ofList ([ [ 0f ]; [ 1f ]; [ 1f ]; [ 0f ] ], Cpu)

let l1 = Linear.init 2 16 F32 Cpu
let l2 = Linear.init 16 1 F32 Cpu
let model = sequential { l1; Relu; l2 }

let state = Model.state model
let opt = AdamW.createWithLr 0.01 (ModelState.trainableParams state)

for epoch in 1..500 do
    scoped {
        opt.zeroGrad ()
        let pred = model.forward x
        let loss = Loss.mse pred y
        loss.backward ()
        opt.step ()
    }
```

## Features

- **Layers** -- Linear, Conv1d/Conv2d, Embedding, BatchNorm, LayerNorm, GroupNorm, Dropout, pooling, activations
- **Blocks** -- LSTM, GRU, MultiHeadAttention, PreNormTransformerBlock, PostNormTransformerBlock
- **Composition** -- `sequential { }` and `pipeline { }` CEs
- **Training** -- SGD, AdamW, learning-rate schedulers, loss functions, metrics, gradient clipping, checkpoints

## License

[MIT](https://github.com/toro-ml/toro/blob/main/LICENSE)
