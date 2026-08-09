module GraphTests

open Xunit
open FsUnit.Xunit
open Toro
open Toro.GNN
open TestHelper

let private mkEdgeIndex (data: int64 array2d) =
    Tensor.ofTorchTensor (TorchSharp.torch.tensor data)
    |> unwrap

[<Fact>]
let ``addSelfLoops appends diagonal edges`` () =
    // Graph: 0 -> 1, 1 -> 2 (2 edges, 3 nodes)
    let edgeIndex = mkEdgeIndex (array2D [| [| 0L; 1L |]; [| 1L; 2L |] |])

    let result = GraphUtils.addSelfLoops edgeIndex 3 |> unwrap
    // Original 2 edges + 3 self-loops = 5
    result.Shape |> should equal [ 2; 5 ]

[<Fact>]
let ``degree computes node degrees`` () =
    // Targets: [1, 0, 2, 1] -> deg(0)=1, deg(1)=2, deg(2)=1
    let index =
        Tensor.ofTorchTensor (TorchSharp.torch.tensor ([| 1L; 0L; 2L; 1L |]: int64 array))
        |> unwrap

    let deg = GraphUtils.degree index 3 F32 Cpu |> unwrap
    deg.Shape |> should equal [ 3 ]

    let d0 = deg[0] |> scalarF32
    let d1 = deg[1] |> scalarF32
    let d2 = deg[2] |> scalarF32
    d0 |> should (equalWithin 1e-5) 1.0f
    d1 |> should (equalWithin 1e-5) 2.0f
    d2 |> should (equalWithin 1e-5) 1.0f

[<Fact>]
let ``GCNConv forward produces correct shape`` () =
    let conv = GCNConv.init 4 8 F32 Cpu |> unwrap

    // 3 nodes, 4 features each
    let x = Tensor.randn ([ 3; 4 ], F32, Cpu) |> unwrap

    // Undirected edges: 0-1, 1-2
    let edgeIndex =
        mkEdgeIndex (array2D [| [| 0L; 1L; 1L; 2L |]; [| 1L; 0L; 2L; 1L |] |])

    let out = conv.forward (x, edgeIndex) |> unwrap
    out.Shape |> should equal [ 3; 8 ]

[<Fact>]
let ``GCNConv no bias forward works`` () =
    let conv = GCNConv.initNoBias 4 8 F32 Cpu |> unwrap

    let x = Tensor.randn ([ 3; 4 ], F32, Cpu) |> unwrap

    let edgeIndex = mkEdgeIndex (array2D [| [| 0L; 1L |]; [| 1L; 0L |] |])

    let out = conv.forward (x, edgeIndex) |> unwrap
    out.Shape |> should equal [ 3; 8 ]
    conv.Bias |> should equal None

[<Fact>]
let ``GCNConv output has gradients`` () =
    let conv = GCNConv.init 4 8 F32 Cpu |> unwrap

    let x = Tensor.randn ([ 3; 4 ], F32, Cpu) |> unwrap

    let edgeIndex =
        mkEdgeIndex (array2D [| [| 0L; 1L; 1L; 2L |]; [| 1L; 0L; 2L; 1L |] |])

    let out = conv.forward (x, edgeIndex) |> unwrap
    let loss = out.sumAll () |> unwrap
    loss.backward () |> unwrap
    conv.Weight.RequiresGrad |> should equal true

[<Fact>]
let ``GraphData create sets fields`` () =
    let x = Tensor.randn ([ 5; 3 ], F32, Cpu) |> unwrap

    let ei = mkEdgeIndex (array2D [| [| 0L; 1L |]; [| 1L; 2L |] |])

    let g = GraphData.create x ei
    GraphData.numNodes g |> should equal 5
    GraphData.numEdges g |> should equal 2
    g.EdgeAttr |> should equal None
    g.Batch |> should equal None
