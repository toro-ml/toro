---
title: Tensor
category: Documentation
categoryindex: 1
index: 3
---

# Tensor

A `Tensor` is a multi-dimensional array. Toro wraps TorchSharp tensors with a safe F# API.
Most tensor methods return `Result<Tensor, ToroError>`.
Arithmetic operators (`+`, `-`, `*`, `/`) throw exceptions directly and return `Tensor`.
See [Core Concepts](concepts.html) for the full operator return type rules.

## Properties

| Property | Type | Description |
| --- | --- | --- |
| `t.Shape` | `int list` | Shape of the tensor |
| `t.Rank` | `int` | Number of dimensions |
| `t.DType` | `DType` | Data type of elements |
| `t.Device` | `Device` | Storage device |
| `t.ElemCount` | `int64` | Total number of elements |
| `t.IsContiguous` | `bool` | Whether the tensor is contiguous in memory |
| `t.RequiresGrad` | `bool` | Whether gradient tracking is enabled |

## Factory Methods

All factory methods return `Result<Tensor, ToroError>`.

```fsharp
open Toro

result {
    let! z = Tensor.zeros ([ 2; 3 ], F32, Cpu)
    let! o = Tensor.ones ([ 2; 3 ], F32, Cpu)
    let! r = Tensor.rand ([ 2; 3 ], F32, Cpu)
    let! n = Tensor.randn ([ 2; 3 ], F32, Cpu)
    let! f = Tensor.full ([ 2; 3 ], 3.14, F32, Cpu)
    let! a = Tensor.arange (5.0, F32, Cpu)
    let! a2 = Tensor.arange (1.0, 10.0, F32, Cpu)
}
```

Create tensors from F# arrays:

```fsharp
result {
    let! t = Tensor.ofFloat32Array [| 1f; 2f; 3f |] Cpu
    let! t2 = Tensor.ofFloat32Array2D ([| [| 1f; 2f |]; [| 3f; 4f |] |], Cpu)
}
```

### Combining Tensors

| Method | Description |
| --- | --- |
| `Tensor.cat (tensors, dim)` | Concatenate along a dimension |
| `Tensor.stack (tensors, dim)` | Stack along a new dimension |
| `Tensor.where (cond, x, y)` | Element-wise conditional select |

```fsharp
result {
    let! a = Tensor.ones ([ 2; 3 ], F32, Cpu)
    let! b = Tensor.zeros ([ 2; 3 ], F32, Cpu)
    let! catted = Tensor.cat ([ a; b ], 0)     // shape: [4; 3]
    let! stacked = Tensor.stack ([ a; b ], 0)  // shape: [2; 2; 3]
}
```

### Special Tensors

| Method | Description |
| --- | --- |
| `Tensor.causalMask (seqLen, dtype, device)` | Upper-triangular mask filled with `-infinity` |
| `Tensor.ofTorchTensor t` | Wrap an existing `torch.Tensor` |

## Arithmetic Operators

Arithmetic operators return `Tensor` directly. They throw exceptions on error.

```fsharp
let c = a + b
let d = a - b
let e = a * b
let f = a / b
let g = -a
```

Scalar operations:

```fsharp
let s = t + 1.0
let s2 = t * 2.0
let s3 = t - 0.5
let s4 = t / 3.0
```

## Shape Operations

| Method | Description |
| --- | --- |
| `t.reshape dims` | Reshape the tensor |
| `t.view dims` | View with a new shape (must be contiguous) |
| `t.flatten (startDim, endDim)` | Flatten a range of dimensions |
| `t.flattenAll ()` | Flatten all dimensions |
| `t.squeeze dim` | Remove a dimension of size 1 |
| `t.unsqueeze dim` | Insert a dimension of size 1 |
| `t.t ()` | Transpose a 2D tensor |
| `t.transpose (d0, d1)` | Swap two dimensions |
| `t.permute dims` | Reorder all dimensions |
| `t.expand shape` | Expand to a larger shape (broadcast) |
| `t.broadcastLeft shape` | Broadcast to a target shape |
| `t.repeatInterleave (repeats, dim)` | Repeat elements along a dimension |
| `t.pad (padding, value)` | Pad with a constant value |
| `t.tril (?diagonal)` | Lower-triangular part |
| `t.triu (?diagonal)` | Upper-triangular part |
| `t.contiguous ()` | Make the tensor contiguous in memory |
| `t.dim d` | Get the size of dimension `d` |

All methods above return `Result`.

```fsharp
result {
    let! t = Tensor.randn ([ 2; 3; 4 ], F32, Cpu)
    let! reshaped = t.reshape [ 6; 4 ]
    let! transposed = t.transpose (0, 2)
    let! permuted = t.permute [ 2; 0; 1 ]
    let! flat = t.flatten (1, -1)
}
```

## Indexing

### Simple Indexing

Use the `Item` property or `GetSlice` for basic indexing.
These throw on error and return `Tensor`:

```fsharp
let row = t[0]          // first row
let elem = t[1, 2]      // element at [1, 2]
let slice = t[0..2]     // rows 0 to 2 (inclusive)
let byTensor = t[idx]   // index with a tensor
```

### Advanced Indexing with `at`

Use the `at` method with `TIdx` for complex indexing patterns.
`at` throws on error and returns `Tensor`:

```fsharp
let result = t.at [ I 1; S(0, 3) ]
```

The `TIdx` discriminated union has these cases:

| Case | Description | Example |
| --- | --- | --- |
| `I n` | Select a single index | `I 0` |
| `S(start, stop)` | Slice from start to stop (exclusive) | `S(0, 3)` |
| `Sf start` | Slice from start to the end | `Sf 2` |
| `St stop` | Slice from the beginning to stop (exclusive) | `St 3` |
| `A` | Select all (`:`) | `A` |
| `T tensor` | Index with a tensor | `T indices` |
| `E` | Ellipsis (`...`) | `E` |
| `N` | Insert a new dimension (`None`) | `N` |

### Other Indexing Methods

These methods return `Result`:

| Method | Description |
| --- | --- |
| `t.indexSelect (dim, index)` | Select elements by index tensor along a dimension |
| `t.gather (dim, index)` | Gather elements along a dimension |
| `t.narrow (dim, start, length)` | Narrow a dimension to a sub-range |
| `t.chunk (n, dim)` | Split into `n` chunks along a dimension |
| `t.maskedFill (mask, value)` | Fill elements where mask is true |
| `t.oneHot numClasses` | One-hot encode (input must be integer type) |

## Comparison

Element-wise comparison operators return a boolean tensor. They throw on error:

```fsharp
let eq = a .=. b    // equal
let ne = a .<>. b   // not equal
let gt = a .>. b    // greater than
let lt = a .<. b    // less than
let ge = a .>=. b   // greater or equal
let le = a .<=. b   // less or equal
```

Instance methods (also throw on error):

| Method | Scalar variant |
| --- | --- |
| `t.eq other` | `t.eqScalar s` |
| `t.ne other` | `t.neScalar s` |
| `t.gt other` | `t.gtScalar s` |
| `t.lt other` | `t.ltScalar s` |
| `t.ge other` | `t.geScalar s` |
| `t.le other` | `t.leScalar s` |

## Reduction

All reduction methods return `Result`:

| Method | Description |
| --- | --- |
| `t.sumAll ()` | Sum all elements |
| `t.sum (dim, ?keepDim)` | Sum along a dimension |
| `t.meanAll ()` | Mean of all elements |
| `t.mean (dim, ?keepDim)` | Mean along a dimension |
| `t.argmax (dim, ?keepDim)` | Index of the maximum along a dimension |
| `t.argmin (dim, ?keepDim)` | Index of the minimum along a dimension |
| `t.max (dim, ?keepDim)` | Maximum values and indices along a dimension |
| `t.min (dim, ?keepDim)` | Minimum values and indices along a dimension |

## Math Operations

All math methods return `Result`:

| Method | Description |
| --- | --- |
| `t.matmul other` | Matrix multiplication |
| `t.exp ()` | Element-wise exponential |
| `t.log ()` | Element-wise natural log |
| `t.sqrt ()` | Element-wise square root |
| `t.sqr ()` | Element-wise square |
| `t.pow exponent` | Element-wise power |
| `t.abs ()` | Element-wise absolute value |
| `t.neg ()` | Element-wise negation |
| `t.clamp (min, max)` | Clamp values to a range |
| `t.affine (mul, add)` | Linear transform: `x * mul + add` |

### Activation Functions

| Method | Description |
| --- | --- |
| `t.relu ()` | ReLU |
| `t.gelu ()` | GELU |
| `t.silu ()` | SiLU (Swish) |
| `t.tanh ()` | Tanh |
| `t.sigmoid ()` | Sigmoid |
| `t.leakyRelu slope` | Leaky ReLU |
| `t.elu alpha` | ELU |
| `t.mish ()` | Mish |
| `t.softmax dim` | Softmax along a dimension |
| `t.logSoftmax dim` | Log-softmax along a dimension |
| `t.dropout (p, train)` | Dropout (functional) |

## Scalar Extraction

Extract a scalar value from a 0-dimensional tensor:

| Method | Return type |
| --- | --- |
| `t.item ()` | `float` (throws) |
| `t.itemF32 ()` | `float32` (throws) |
| `t.itemI64 ()` | `int64` (throws) |
| `t.itemI32 ()` | `int` (throws) |
| `t.toFloat32Scalar ()` | `Result<float32, ToroError>` |
| `t.toFloat64Scalar ()` | `Result<float, ToroError>` |
| `t.toInt32Scalar ()` | `Result<int, ToroError>` |
| `t.toInt64Scalar ()` | `Result<int64, ToroError>` |

## Conversion

| Method | Description |
| --- | --- |
| `t.toDevice device` | Move to a different device |
| `t.toDType dtype` | Cast to a different data type |
| `t.detach ()` | Detach from the computation graph |
| `t.clone ()` | Create a deep copy |

All methods above return `Result`.

## Autograd

| Method | Return type | Description |
| --- | --- | --- |
| `t.requiresGrad (?rg)` | `Result<Tensor>` | Enable gradient tracking |
| `t.backward ()` | `Result<unit>` | Compute gradients |
| `t.grad ()` | `Result<Tensor>` | Get the gradient tensor |
| `t.zeroGrad ()` | `unit` | Zero out gradients |
| `t.copyInPlace src` | `Result<unit>` | Copy data from another tensor (no grad) |

```fsharp
result {
    let! w = Tensor.randn ([ 3; 1 ], F32, Cpu)
    let! w = w.requiresGrad ()
    // ... forward pass and loss computation ...
    do! loss.backward ()
    let! g = w.grad ()
}
```

## Persistence

| Method | Description |
| --- | --- |
| `t.save path` | Save tensor to a file |
| `Tensor.load path` | Load tensor from a file |

```fsharp
result {
    let! t = Tensor.randn ([ 3; 4 ], F32, Cpu)
    do! t.save "weights.pt"
    let! loaded = Tensor.load "weights.pt"
}
```
