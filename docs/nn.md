---
title: Neural Networks
category: Documentation
categoryindex: 1
index: 4
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

- **`IModule`** -- `forward: Tensor -> Result<Tensor, ToroError>`. Use for layers that behave the same during training and inference.
- **`IModuleT`** -- `forwardT: Tensor -> train: bool -> Result<Tensor, ToroError>`. Use for layers like Dropout and BatchNorm that change behavior based on the training phase.

## Layers

### Linear

A fully connected layer. Implements `IModule`:

```fsharp
result {
    let! l = Linear.init 784 256 F32 Cpu
    let! out = l.forward input
}
```

`Linear.initNoBias` creates a linear layer without a bias term.

### Conv1d

A 1D convolution layer. Implements `IModule`:

```fsharp
result {
    let! conv = Conv1d.initDefault 3 16 5 F32 Cpu  // inCh=3, outCh=16, kernel=5
    let! out = conv.forward input
}
```

Configure with `Conv1dConfig`:

```fsharp
result {
    let! conv = Conv1d.init 3 16 5 { Conv1dConfig.defaultConfig with Padding = 2; Stride = 2 } F32 Cpu
}
```

`Conv1d.initNoBias` creates a layer without a bias term.

### Conv2d

A 2D convolution layer. Implements `IModule`:

```fsharp
result {
    let! conv = Conv2d.init { Conv2dConfig.defaultConfig with InChannels = 1; OutChannels = 32; KernelSize = 3 } F32 Cpu
    let! out = conv.forward input
}
```

`Conv2dConfig` has the same fields as `Conv1dConfig`: `Padding`, `Stride`, `Dilation`, `Groups`.

### Pooling

Pooling layers reduce spatial dimensions. All implement `IModule`:

| Layer | Factory | Description |
| --- | --- | --- |
| `MaxPool1d` | `MaxPool1d.createDefault kernelSize` | 1D max pooling |
| `MaxPool2d` | `MaxPool2d.createDefault kernelSize` | 2D max pooling |
| `AvgPool2d` | `AvgPool2d.createDefault kernelSize` | 2D average pooling |

Use `create` for full control:

```fsharp
let pool = MaxPool2d.create 3 2 1  // kernelSize=3, stride=2, padding=1
```

### Embedding

A lookup table that maps integer indices to dense vectors. Implements `IModule`:

```fsharp
result {
    let! emb = Embedding.init 10000 128 F32 Cpu
    let! out = emb.forward indices  // indices: int64 tensor
}
```

### Normalization

#### LayerNorm

Layer normalization. Implements `IModule`:

```fsharp
result {
    let! ln = LayerNorm.initDefault 128 F32 Cpu
    let! out = ln.forward input
}
```

Configure with `LayerNormConfig` (fields: `Eps`, `RemoveMean`, `Affine`).

#### RmsNorm

Root mean square normalization. Implements `IModule`:

```fsharp
result {
    let! rn = RmsNorm.init 128 1e-5 F32 Cpu
    let! out = rn.forward input
}
```

#### BatchNorm

Batch normalization. Implements `IModuleT` (behavior differs between training and inference):

```fsharp
result {
    let! bn = BatchNorm.initDefault 64 F32 Cpu
    let! out = bn.forwardT input true   // training
    let! out = bn.forwardT input false  // inference
}
```

Configure with `BatchNormConfig` (fields: `Eps`, `Momentum`, `Affine`).

#### GroupNorm

Group normalization. Implements `IModule`:

```fsharp
result {
    let! gn = GroupNorm.initDefault 8 64 F32 Cpu  // 8 groups, 64 channels
    let! out = gn.forward input
}
```

### Dropout

Randomly zeroes elements during training. Implements `IModuleT`:

```fsharp
let drop = Dropout.create 0.5
```

### Activation

Activation functions implement `IModule`:

| Case | Description |
| --- | --- |
| `Relu` | ReLU |
| `Gelu` | GELU |
| `Silu` | SiLU (Swish) |
| `Tanh` | Tanh |
| `Sigmoid` | Sigmoid |
| `LeakyRelu slope` | Leaky ReLU |
| `Elu alpha` | ELU |
| `Mish` | Mish |

Use them in a sequential model or call `forward` directly:

```fsharp
let! out = Relu.forward input
```

### Func, FuncT, and Identity

Wrap any function as a module:

```fsharp
let flatten = Func.create _.flatten(1, -1)
let dropFn = FuncT.create (fun x train -> drop.forwardT x train)
```

`Identity()` is a module that returns its input unchanged:

```fsharp
let skip = Identity()
```

## Recurrent Layers

### IRNN Interface

All recurrent layers implement `IRNN<'State>`:

```fsharp
type IRNN<'State> =
    abstract zeroState: batchDim: int -> Result<'State, ToroError>
    abstract step: Tensor -> 'State -> Result<'State, ToroError>
    abstract seq: Tensor -> Result<'State list, ToroError>
    abstract statesToTensor: 'State list -> Result<Tensor, ToroError>
```

### LSTM

Long Short-Term Memory. State type: `LSTMState = { H: Tensor; C: Tensor }`:

```fsharp
result {
    let! lstm = LSTM.initDefault 128 256 F32 Cpu
    let! state0 = lstm.zeroState batchSize
    let! state1 = lstm.step input state0        // one step
    let! states = lstm.seq inputSequence         // full sequence
    let! output = lstm.statesToTensor states     // [seqLen; batch; hidden]
}
```

Configure with `LSTMConfig` for custom weight initialization.

### GRU

Gated Recurrent Unit. State type: `GRUState = { H: Tensor }`:

```fsharp
result {
    let! gru = GRU.initDefault 128 256 F32 Cpu
    let! state0 = gru.zeroState batchSize
    let! states = gru.seq inputSequence
}
```

## Attention and Transformer

### MultiHeadAttention

Multi-head self-attention. Implements `IModule`:

```fsharp
result {
    let! attn = MultiHeadAttention.init 128 4 F32 Cpu  // dim=128, 4 heads
    let! out = attn.forward (input, ?mask = mask)
}
```

Supports an optional `KvCache` for autoregressive generation.

### TransformerBlock

A pre-norm transformer block with multi-head attention and a feed-forward network. Implements `IModule`:

```fsharp
result {
    let! block = TransformerBlock.init 128 4 512 F32 Cpu  // dim=128, 4 heads, ffDim=512
    let! out = block.forward (input, ?mask = mask)
}
```

### KvCache

Key-value cache for autoregressive inference:

```fsharp
let cache = KvCache.create 2  // cache dim (typically the sequence dimension)
let! k, v = cache.append (newK, newV)
cache.reset ()
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
    l1
    Relu
    drop
    l2
}

let! trainOut = model.forwardT input true
let! evalOut = model.forwardT input false
```

## Weight Initialization

The `Init` type controls how layer weights are initialized:

| Case | Description |
| --- | --- |
| `Const v` | Fill with constant value `v` |
| `Randn (mean, stdev)` | Normal distribution |
| `Uniform (lo, up)` | Uniform distribution |
| `KaimingNormal` | Kaiming normal (default for most layers) |

Use `Init.toTensor` to create a tensor, or `Init.toParam` to create a tensor with `requiresGrad`:

```fsharp
result {
    let! w = Init.toParam [ 256; 128 ] F32 Cpu KaimingNormal
}
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

`Model.namedParams` returns all parameters with their names:

```fsharp
let params = Model.namedParams myModel
// [("Layer1.Weight", tensor); ("Layer1.Bias", tensor); ...]
```

## Model Persistence

Save and load model parameters:

```fsharp
result {
    do! Model.save myModel "checkpoints/epoch10"
    do! Model.loadInto myModel "checkpoints/epoch10"
}
```

`Model.save` writes each parameter as a separate file under the directory.
`Model.loadInto` loads parameters from files and copies them into the model tensors.
