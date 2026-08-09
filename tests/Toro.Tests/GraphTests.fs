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

// --- MessagePassing tests ---

[<Fact>]
let ``MessagePassing aggregate Add scatters correctly`` () =
    // 2 edges targeting nodes 0 and 1; 3 nodes total; 2 features
    let msg =
        Tensor.ofArray (array2D [| [| 1.0f; 2.0f |]; [| 3.0f; 4.0f |] |], Cpu)
        |> unwrap

    let targetIdx =
        Tensor.ofTorchTensor (TorchSharp.torch.tensor ([| 0L; 1L |]: int64 array))
        |> unwrap

    let out = MessagePassing.aggregate Add msg targetIdx 3 2 |> unwrap
    out.Shape |> should equal [ 3; 2 ]

    out.at [ I 0; I 0 ]
    |> scalarF32
    |> should (equalWithin 1e-5) 1.0f

    out.at [ I 0; I 1 ]
    |> scalarF32
    |> should (equalWithin 1e-5) 2.0f

    out.at [ I 1; I 0 ]
    |> scalarF32
    |> should (equalWithin 1e-5) 3.0f

    out.at [ I 1; I 1 ]
    |> scalarF32
    |> should (equalWithin 1e-5) 4.0f

[<Fact>]
let ``MessagePassing aggregate Mean divides by count`` () =
    // 3 edges: two target node 0, one targets node 1
    let msg =
        Tensor.ofArray (array2D [| [| 2.0f; 4.0f |]; [| 4.0f; 6.0f |]; [| 1.0f; 1.0f |] |], Cpu)
        |> unwrap

    let targetIdx =
        Tensor.ofTorchTensor (TorchSharp.torch.tensor ([| 0L; 0L; 1L |]: int64 array))
        |> unwrap

    let out = MessagePassing.aggregate Mean msg targetIdx 2 2 |> unwrap
    out.Shape |> should equal [ 2; 2 ]
    // Node 0: mean([2,4],[4,6]) = [3,5]
    let v00 = out.at [ I 0; I 0 ] |> scalarF32
    let v01 = out.at [ I 0; I 1 ] |> scalarF32
    v00 |> should (equalWithin 1e-5) 3.0f
    v01 |> should (equalWithin 1e-5) 5.0f

[<Fact>]
let ``MessagePassing edgeSoftmax sums to 1 per target`` () =
    // 4 edges: edges 0,1 target node 0; edges 2,3 target node 1
    let scores =
        Tensor.ofTorchTensor (TorchSharp.torch.tensor ([| 1.0f; 2.0f; 0.5f; 0.5f |]: float32 array))
        |> unwrap

    let targetIdx =
        Tensor.ofTorchTensor (TorchSharp.torch.tensor ([| 0L; 0L; 1L; 1L |]: int64 array))
        |> unwrap

    let attn = MessagePassing.edgeSoftmax scores targetIdx 2 |> unwrap
    attn.Shape |> should equal [ 4 ]
    // Sum of edges targeting node 0 should be ~1
    let a0 = attn[0] |> scalarF32
    let a1 = attn[1] |> scalarF32
    (a0 + a1) |> should (equalWithin 1e-5) 1.0f
    // Sum of edges targeting node 1 should be ~1
    let a2 = attn[2] |> scalarF32
    let a3 = attn[3] |> scalarF32
    (a2 + a3) |> should (equalWithin 1e-5) 1.0f

// --- GATConv tests ---

[<Fact>]
let ``GATConv single-head forward produces correct shape`` () =
    let conv = GATConv.initDefault 4 8 F32 Cpu |> unwrap

    let x = Tensor.randn ([ 3; 4 ], F32, Cpu) |> unwrap

    let edgeIndex =
        mkEdgeIndex (array2D [| [| 0L; 1L; 1L; 2L |]; [| 1L; 0L; 2L; 1L |] |])

    let out = conv.forward (x, edgeIndex) |> unwrap
    out.Shape |> should equal [ 3; 8 ]

[<Fact>]
let ``GATConv multi-head concat produces correct shape`` () =
    let conv = GATConv.init 4 8 4 true 0.2 F32 Cpu |> unwrap

    let x = Tensor.randn ([ 5; 4 ], F32, Cpu) |> unwrap

    let edgeIndex =
        mkEdgeIndex (array2D [| [| 0L; 1L; 2L; 3L; 4L |]; [| 1L; 2L; 3L; 4L; 0L |] |])

    let out = conv.forward (x, edgeIndex) |> unwrap
    // concat=true: heads * outChannels = 4 * 8 = 32
    out.Shape |> should equal [ 5; 32 ]

[<Fact>]
let ``GATConv multi-head mean produces correct shape`` () =
    let conv = GATConv.init 4 8 4 false 0.2 F32 Cpu |> unwrap

    let x = Tensor.randn ([ 5; 4 ], F32, Cpu) |> unwrap

    let edgeIndex =
        mkEdgeIndex (array2D [| [| 0L; 1L; 2L; 3L; 4L |]; [| 1L; 2L; 3L; 4L; 0L |] |])

    let out = conv.forward (x, edgeIndex) |> unwrap
    // concat=false: outChannels = 8
    out.Shape |> should equal [ 5; 8 ]

[<Fact>]
let ``GATConv output has gradients`` () =
    let conv = GATConv.initDefault 4 8 F32 Cpu |> unwrap

    let x = Tensor.randn ([ 3; 4 ], F32, Cpu) |> unwrap

    let edgeIndex =
        mkEdgeIndex (array2D [| [| 0L; 1L; 1L; 2L |]; [| 1L; 0L; 2L; 1L |] |])

    let out = conv.forward (x, edgeIndex) |> unwrap
    let loss = out.sumAll () |> unwrap
    loss.backward () |> unwrap
    conv.Weight.RequiresGrad |> should equal true
    conv.AttSrc.RequiresGrad |> should equal true

// --- SAGEConv tests ---

[<Fact>]
let ``SAGEConv forward produces correct shape`` () =
    let conv = SAGEConv.init 4 8 F32 Cpu |> unwrap

    let x = Tensor.randn ([ 3; 4 ], F32, Cpu) |> unwrap

    let edgeIndex =
        mkEdgeIndex (array2D [| [| 0L; 1L; 1L; 2L |]; [| 1L; 0L; 2L; 1L |] |])

    let out = conv.forward (x, edgeIndex) |> unwrap
    out.Shape |> should equal [ 3; 8 ]

[<Fact>]
let ``SAGEConv no bias forward works`` () =
    let conv = SAGEConv.initNoBias 4 8 F32 Cpu |> unwrap

    let x = Tensor.randn ([ 3; 4 ], F32, Cpu) |> unwrap

    let edgeIndex = mkEdgeIndex (array2D [| [| 0L; 1L |]; [| 1L; 0L |] |])

    let out = conv.forward (x, edgeIndex) |> unwrap
    out.Shape |> should equal [ 3; 8 ]
    conv.Bias |> should equal None

[<Fact>]
let ``SAGEConv output has gradients`` () =
    let conv = SAGEConv.init 4 8 F32 Cpu |> unwrap

    let x = Tensor.randn ([ 3; 4 ], F32, Cpu) |> unwrap

    let edgeIndex =
        mkEdgeIndex (array2D [| [| 0L; 1L; 1L; 2L |]; [| 1L; 0L; 2L; 1L |] |])

    let out = conv.forward (x, edgeIndex) |> unwrap
    let loss = out.sumAll () |> unwrap
    loss.backward () |> unwrap
    conv.WeightSelf.RequiresGrad |> should equal true
    conv.WeightNeighbor.RequiresGrad |> should equal true
