---
title: Tensor
category: Documentation
categoryindex: 1
index: 2
---

# Tensor

A `Tensor` is a multi-dimensional array. Toro wraps TorchSharp tensors with a safe F# API.
Most tensor operations return `Result<Tensor, ToroError>`.
Arithmetic operators (`+`, `-`, `*`, `/`) throw exceptions directly and return `Tensor`.

## Create Tensors

Use factory methods to create tensors. Each method requires a shape, dtype, and device.

```fsharp
open Toro

result {
    let! z = Tensor.zeros ([ 2; 3 ], F32, Cpu)
    let! o = Tensor.ones ([ 2; 3 ], F32, Cpu)
    let! r = Tensor.rand ([ 2; 3 ], F32, Cpu)
    let! n = Tensor.randn ([ 2; 3 ], F32, Cpu)
    let! f = Tensor.full ([ 2; 3 ], 3.14, F32, Cpu)
    let! a = Tensor.arange (5.0, F32, Cpu)
}
```

Create tensors from F# arrays:

```fsharp
result {
    let! t = Tensor.ofFloat32Array [| 1f; 2f; 3f |] Cpu
    let! t2 = Tensor.ofFloat32Array2D ([| [| 1f; 2f |]; [| 3f; 4f |] |], Cpu)
}
```

## Arithmetic Operators

Arithmetic operators return `Tensor` directly. They throw exceptions on error.

```fsharp
let c = a + b
let d = a - b
let e = a * b
let f = a / b
let g = -a
```

Scalar operations are also available:

```fsharp
let s = t + 1.0
let s2 = t * 2.0
let s3 = t - 0.5
let s4 = t / 3.0
```

## Shape Operations

| Method | Return type | Description |
| --- | --- | --- |
| `t.Shape` | `int list` | Get the shape |
| `t.Rank` | `int` | Get the number of dimensions |
| `t.reshape dims` | `Result` | Reshape the tensor |
| `t.view dims` | `Result` | View with a new shape |
| `t.flatten (startDim, endDim)` | `Result` | Flatten dimensions |
| `t.squeeze dim` | `Result` | Remove a dimension of size 1 |
| `t.unsqueeze dim` | `Result` | Add a dimension of size 1 |
| `t.transpose (d0, d1)` | `Result` | Swap two dimensions |
| `t.permute dims` | `Result` | Reorder dimensions |
| `t.expand shape` | `Result` | Expand to a larger shape |
| `t.contiguous ()` | `Result` | Make the tensor contiguous in memory |

Example:

```fsharp
result {
    let! t = Tensor.randn ([ 2; 3; 4 ], F32, Cpu)
    let! reshaped = t.reshape [ 6; 4 ]
    let! transposed = t.transpose (0, 2)
    let! permuted = t.permute [| 2; 0; 1 |]
}
```

## Indexing

### Simple Indexing

Use the `Item` property or `GetSlice` for basic indexing:

```fsharp
let row = t[0]          // first row
let elem = t[1, 2]      // element at [1, 2]
let slice = t[0..2]     // rows 0 to 2 (inclusive)
```

### Advanced Indexing with `at`

Use the `at` method with `TIdx` for complex indexing patterns:

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

Example:

```fsharp
result {
    let! t = Tensor.randn ([ 4; 5; 6 ], F32, Cpu)
    let row = t.at [ I 2 ]
    let sub = t.at [ A; S(1, 3) ]
    let first3 = t.at [ St 3 ]
}
```

## Comparison

Element-wise comparison operators return a boolean tensor:

```fsharp
let eq = a .=. b    // element-wise equal
let ne = a .<>. b   // element-wise not equal
let gt = a .>. b    // element-wise greater than
let lt = a .<. b    // element-wise less than
let ge = a .>=. b   // element-wise greater or equal
let le = a .<=. b   // element-wise less or equal
```

Scalar comparison methods:

```fsharp
let mask = t.gtScalar 0.5
let eq0 = t.eqScalar 0.0
```

## Reduction

| Method | Description |
| --- | --- |
| `t.sumAll ()` | Sum all elements |
| `t.meanAll ()` | Mean of all elements |
| `t.sum dim` | Sum along a dimension |
| `t.mean dim` | Mean along a dimension |
| `t.argmax dim` | Index of the maximum along a dimension |
| `t.max ()` | Maximum element |
| `t.min ()` | Minimum element |

## Math Operations

| Method | Description |
| --- | --- |
| `t.matmul other` | Matrix multiplication |
| `t.exp ()` | Element-wise exponential |
| `t.log ()` | Element-wise natural log |
| `t.sqrt ()` | Element-wise square root |
| `t.abs ()` | Element-wise absolute value |
| `t.relu ()` | Element-wise ReLU |
| `t.softmax dim` | Softmax along a dimension |
| `t.logSoftmax dim` | Log-softmax along a dimension |
| `t.tanh ()` | Element-wise tanh |
| `t.sigmoid ()` | Element-wise sigmoid |

## Scalar Extraction

Use `item()` to get a scalar value from a 0-dimensional tensor:

```fsharp
let value: float32 = loss.item ()
```

## Gradient

```fsharp
result {
    let! w = Tensor.randn ([ 3; 1 ], F32, Cpu)
    let! w = w.requiresGrad ()

    // ... forward pass and loss computation ...

    do! loss.backward ()
    let! grad = w.grad ()
}
```
