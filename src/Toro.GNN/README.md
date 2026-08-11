# Toro.GNN

[![Toro.GNN](https://img.shields.io/nuget/v/Toro.GNN.svg?label=Toro.GNN)](https://www.nuget.org/packages/Toro.GNN)

Graph neural network layers for [Toro](https://www.nuget.org/packages/Toro). Message-passing convolutions, graph data, batching, and pooling.

**[Documentation](https://toro-ml.github.io/toro/)**

## Installation

```bash
dotnet add package Toro.GNN
dotnet add package TorchSharp-cpu
```

## Quick Example

```fsharp
open Toro
open Toro.NN
open Toro.GNN

let x = Tensor.randn ([ 4; 16 ], F32, Cpu)
let edgeIndex = Tensor.ofList ([ [ 0L; 1L; 2L; 3L ]; [ 1L; 2L; 3L; 0L ] ], Cpu)
let g = GraphData.create x edgeIndex

let conv = GCNConv.init 16 32 F32 Cpu
let out = conv.forward (g.X, g.EdgeIndex)
```

## Features

- **Graph data** -- COO edge index, node features, multi-graph batching
- **Convolutions** -- GCNConv, GATConv, SAGEConv, GINConv on a shared message-passing base
- **Pooling and norm** -- global mean/max/add pool, GraphNorm

## License

[MIT](https://github.com/toro-ml/toro/blob/main/LICENSE)
