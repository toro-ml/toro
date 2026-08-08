---
title: Neural Networks
category: Documentation
categoryindex: 1
index: 3
---

# Neural Networks

Toro.NN provides layers, modules, and composition tools for neural networks.
Open the namespace to get started:

```fsharp
open Toro
open Toro.NN
```

## Module Interfaces

Toro defines two interfaces for neural network components:

- **`IModule`** -- Has `forward: Tensor -> Result<Tensor, ToroError>`. Use for layers that behave the same during training and inference.
- **`IModuleT`** -- Has `forwardT: Tensor -> train: bool -> Result<Tensor, ToroError>`. Use for layers like Dropout and BatchNorm that change behavior based on the training phase.

## Layers

### Linear

A fully connected layer:

```fsharp
result {
    let! l = Linear.init 784 256 F32 Cpu
    let! out = l.forward input
}
```

`Linear.initNoBias` creates a linear layer without a bias term.

### Conv2d

A 2D convolution layer:

```fsharp
result {
    let! conv = Conv2d.init { Conv2dConfig.defaultConfig with InChannels = 1; OutChannels = 32; KernelSize = 3 } F32 Cpu
    let! out = conv.forward input
}
```

### Embedding

A lookup table for embedding indices into dense vectors:

```fsharp
result {
    let! emb = Embedding.init 10000 128 F32 Cpu
    let! out = emb.forward indices
}
```

### Dropout

Randomly zeroes elements during training. Implements `IModuleT`:

```fsharp
result {
    let drop = Dropout.create 0.5
    let! out = drop.forwardT input true  // training mode
    let! out = drop.forwardT input false // inference mode (no dropout)
}
```

### BatchNorm

Batch normalization. Implements `IModuleT`:

```fsharp
result {
    let! bn = BatchNorm.init 64 F32 Cpu
    let! out = bn.forwardT input true
}
```

### LayerNorm

Layer normalization:

```fsharp
result {
    let! ln = LayerNorm.init 128 F32 Cpu
    let! out = ln.forward input
}
```

### Activation

Activation functions implement `IModule`:

```fsharp
Relu
Gelu
Silu
Tanh
Sigmoid
LeakyRelu 0.2
Elu 1.0
Mish
```

Use them in a sequential model or call `forward` directly:

```fsharp
let! out = Relu.forward input
```

### Func and FuncT

Wrap any function as a module:

```fsharp
let flatten = Func.create _.flatten(1, -1)
let dropoutFn = FuncT.create (fun x train -> drop.forwardT x train)
```

## LSTM and GRU

Recurrent layers for sequence processing:

```fsharp
result {
    let! lstm = LSTM.initDefault 128 256 F32 Cpu
    let! state0 = lstm.zeroState batchSize

    // Process one step
    let! state1 = lstm.step input state0

    // Process a full sequence
    let! states = lstm.seq inputSequence
}
```

## MultiHeadAttention and TransformerBlock

```fsharp
result {
    let! attn = MultiHeadAttention.init 128 4 F32 Cpu
    let! block = TransformerBlock.init 128 4 0.1 F32 Cpu
    let! out = block.forwardT input true
}
```

## Model Composition

### Sequential

Use the `sequential { }` computation expression to compose `IModule` layers:

```fsharp
let! l1 = Linear.init 784 256 F32 Cpu
let! l2 = Linear.init 256 10 F32 Cpu

let model = sequential {
    l1
    Relu
    l2
}

let! output = model.forward input
```

### SequentialT

Use `sequentialT { }` when any layer needs the `train` flag. It accepts both `IModule` and `IModuleT` layers:

```fsharp
let! l1 = Linear.init 784 256 F32 Cpu
let drop = Dropout.create 0.5
let! l2 = Linear.init 256 10 F32 Cpu

let model = sequentialT {
    l1         // IModule -- accepted directly
    Relu       // IModule (Activation)
    drop       // IModuleT
    l2
}

let! trainOut = model.forwardT input true
let! evalOut = model.forwardT input false
```

## Parameter Collection

Use `Model.trainableVars` to collect all trainable parameters from a model record:

```fsharp
type MyModel = {
    Layer1: Linear
    Layer2: Linear
}

let vars = Model.trainableVars myModel
let! opt = AdamW.createWithLr 0.001 vars
```

`Model.trainableVars` uses reflection to find all `Tensor` values in the record.
It traverses nested records and lists (including `Sequential` and `SequentialT`).
