module GraphTests

open Xunit
open FsUnit.Xunit
open Toro
open TorchSharp
open Toro.GNN
open Toro.NN
open TestHelper

let private mkEdgeIndex (data: int64 array2d) = (TorchSharp.torch.tensor data)

[<Fact>]
let ``GNN layers expose explicitly classified named parameters`` () =
    let gcn = GCNConv.init 4 8 torch.float32 torch.CPU

    Model.namedParams gcn
    |> List.map _.Name
    |> should equal [ "Weight"; "Bias" ]

    let gat = GATConv.initDefault 4 8 torch.float32 torch.CPU

    Model.namedParams gat
    |> List.map _.Name
    |> should equal [ "Weight"; "AttSrc"; "AttTgt"; "Bias" ]

    let sage = SAGEConv.init 4 8 torch.float32 torch.CPU

    Model.namedParams sage
    |> List.map _.Name
    |> should equal [ "WeightSelf"; "WeightNeighbor"; "Bias" ]

    let graphNorm = GraphNorm.init 4 torch.float32 torch.CPU

    Model.namedParams graphNorm
    |> List.map _.Name
    |> should equal [ "Gamma"; "Beta"; "Alpha" ]

[<Fact>]
let ``GraphData is rejected as unclassified runtime data`` () =
    let graph =
        GraphData.create (torch.randn ([| 2L; 3L |], dtype = torch.float32)) (mkEdgeIndex (array2D [| [| 0L |]; [| 1L |] |]))

    Assert.Throws<System.InvalidOperationException>(fun () -> Model.namedState graph |> ignore)
    |> ignore


[<Fact>]
let ``addSelfLoops appends diagonal edges`` () =
    // Graph: 0 -> 1, 1 -> 2 (2 edges, 3 nodes)
    let edgeIndex = mkEdgeIndex (array2D [| [| 0L; 1L |]; [| 1L; 2L |] |])

    let result = GraphUtils.addSelfLoops edgeIndex 3
    // Original 2 edges + 3 self-loops = 5
    result.shape |> should equal [| 2L; 5L |]

[<Fact>]
let ``degree computes node degrees`` () =
    // Targets: [1, 0, 2, 1] -> deg(0)=1, deg(1)=2, deg(2)=1
    let index = (TorchSharp.torch.tensor ([| 1L; 0L; 2L; 1L |]: int64 array))


    let deg = GraphUtils.degree index 3 torch.float32 torch.CPU
    deg.shape |> should equal [| 3L |]

    let d0 = deg[0] |> scalarF32
    let d1 = deg[1] |> scalarF32
    let d2 = deg[2] |> scalarF32
    d0 |> should (equalWithin 1e-5) 1.0f
    d1 |> should (equalWithin 1e-5) 2.0f
    d2 |> should (equalWithin 1e-5) 1.0f

[<Fact>]
let ``GCNConv forward produces correct shape`` () =
    let conv = GCNConv.init 4 8 torch.float32 torch.CPU

    // 3 nodes, 4 features each
    let x = torch.randn ([| 3L; 4L |], dtype = torch.float32, device = torch.CPU)

    // Undirected edges: 0-1, 1-2
    let edgeIndex =
        mkEdgeIndex (array2D [| [| 0L; 1L; 1L; 2L |]; [| 1L; 0L; 2L; 1L |] |])

    let out = conv.forward (x, edgeIndex)
    out.shape |> should equal [| 3L; 8L |]

[<Fact>]
let ``GCNConv no bias forward works`` () =
    let conv = GCNConv.initNoBias 4 8 torch.float32 torch.CPU

    let x = torch.randn ([| 3L; 4L |], dtype = torch.float32, device = torch.CPU)

    let edgeIndex = mkEdgeIndex (array2D [| [| 0L; 1L |]; [| 1L; 0L |] |])

    let out = conv.forward (x, edgeIndex)
    out.shape |> should equal [| 3L; 8L |]
    conv.Bias |> should equal None

[<Fact>]
let ``GCNConv output has gradients`` () =
    let conv = GCNConv.init 4 8 torch.float32 torch.CPU

    let x = torch.randn ([| 3L; 4L |], dtype = torch.float32, device = torch.CPU)

    let edgeIndex =
        mkEdgeIndex (array2D [| [| 0L; 1L; 1L; 2L |]; [| 1L; 0L; 2L; 1L |] |])

    let out = conv.forward (x, edgeIndex)
    let loss = out.sum ()
    loss.backward ()
    conv.Weight.requires_grad |> should equal true

[<Fact>]
let ``GraphData create sets fields`` () =
    let x = torch.randn ([| 5L; 3L |], dtype = torch.float32, device = torch.CPU)

    let ei = mkEdgeIndex (array2D [| [| 0L; 1L |]; [| 1L; 2L |] |])

    let g = GraphData.create x ei
    GraphData.numNodes g |> should equal 5L
    GraphData.numEdges g |> should equal 2L
    g.EdgeAttr |> should equal None
    g.Batch |> should equal None

// --- MessagePassing tests ---

[<Fact>]
let ``MessagePassing aggregate Add scatters correctly`` () =
    // 2 edges targeting nodes 0 and 1; 3 nodes total; 2 features
    let msg =
        torch.tensor (array2D [| [| 1.0f; 2.0f |]; [| 3.0f; 4.0f |] |], device = torch.CPU)


    let targetIdx = (TorchSharp.torch.tensor ([| 0L; 1L |]: int64 array))


    let out = MessagePassing.aggregate Add msg targetIdx 3 2
    out.shape |> should equal [| 3L; 2L |]

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
        torch.tensor (array2D [| [| 2.0f; 4.0f |]; [| 4.0f; 6.0f |]; [| 1.0f; 1.0f |] |], device = torch.CPU)


    let targetIdx = (TorchSharp.torch.tensor ([| 0L; 0L; 1L |]: int64 array))


    let out = MessagePassing.aggregate Mean msg targetIdx 2 2
    out.shape |> should equal [| 2L; 2L |]
    // Node 0: mean([2,4],[4,6]) = [3,5]
    let v00 = out.at [ I 0; I 0 ] |> scalarF32
    let v01 = out.at [ I 0; I 1 ] |> scalarF32
    v00 |> should (equalWithin 1e-5) 3.0f
    v01 |> should (equalWithin 1e-5) 5.0f

[<Fact>]
let ``MessagePassing edgeSoftmax sums to 1 per target`` () =
    // 4 edges: edges 0,1 target node 0; edges 2,3 target node 1
    let scores = (TorchSharp.torch.tensor ([| 1.0f; 2.0f; 0.5f; 0.5f |]: float32 array))


    let targetIdx = (TorchSharp.torch.tensor ([| 0L; 0L; 1L; 1L |]: int64 array))


    let attn = MessagePassing.edgeSoftmax scores targetIdx 2
    attn.shape |> should equal [| 4L |]
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
    let conv = GATConv.initDefault 4 8 torch.float32 torch.CPU

    let x = torch.randn ([| 3L; 4L |], dtype = torch.float32, device = torch.CPU)

    let edgeIndex =
        mkEdgeIndex (array2D [| [| 0L; 1L; 1L; 2L |]; [| 1L; 0L; 2L; 1L |] |])

    let out = conv.forward (x, edgeIndex)
    out.shape |> should equal [| 3L; 8L |]

[<Fact>]
let ``GATConv multi-head concat produces correct shape`` () =
    let conv = GATConv.init 4 8 4 true 0.2 torch.float32 torch.CPU

    let x = torch.randn ([| 5L; 4L |], dtype = torch.float32, device = torch.CPU)

    let edgeIndex =
        mkEdgeIndex (array2D [| [| 0L; 1L; 2L; 3L; 4L |]; [| 1L; 2L; 3L; 4L; 0L |] |])

    let out = conv.forward (x, edgeIndex)
    // concat=true: heads * outChannels = 4 * 8 = 32
    out.shape |> should equal [| 5L; 32L |]

[<Fact>]
let ``GATConv multi-head mean produces correct shape`` () =
    let conv = GATConv.init 4 8 4 false 0.2 torch.float32 torch.CPU

    let x = torch.randn ([| 5L; 4L |], dtype = torch.float32, device = torch.CPU)

    let edgeIndex =
        mkEdgeIndex (array2D [| [| 0L; 1L; 2L; 3L; 4L |]; [| 1L; 2L; 3L; 4L; 0L |] |])

    let out = conv.forward (x, edgeIndex)
    // concat=false: outChannels = 8
    out.shape |> should equal [| 5L; 8L |]

[<Fact>]
let ``GATConv output has gradients`` () =
    let conv = GATConv.initDefault 4 8 torch.float32 torch.CPU

    let x = torch.randn ([| 3L; 4L |], dtype = torch.float32, device = torch.CPU)

    let edgeIndex =
        mkEdgeIndex (array2D [| [| 0L; 1L; 1L; 2L |]; [| 1L; 0L; 2L; 1L |] |])

    let out = conv.forward (x, edgeIndex)
    let loss = out.sum ()
    loss.backward ()
    conv.Weight.requires_grad |> should equal true
    conv.AttSrc.requires_grad |> should equal true

// --- SAGEConv tests ---

[<Fact>]
let ``SAGEConv forward produces correct shape`` () =
    let conv = SAGEConv.init 4 8 torch.float32 torch.CPU

    let x = torch.randn ([| 3L; 4L |], dtype = torch.float32, device = torch.CPU)

    let edgeIndex =
        mkEdgeIndex (array2D [| [| 0L; 1L; 1L; 2L |]; [| 1L; 0L; 2L; 1L |] |])

    let out = conv.forward (x, edgeIndex)
    out.shape |> should equal [| 3L; 8L |]

[<Fact>]
let ``SAGEConv no bias forward works`` () =
    let conv = SAGEConv.initNoBias 4 8 torch.float32 torch.CPU

    let x = torch.randn ([| 3L; 4L |], dtype = torch.float32, device = torch.CPU)

    let edgeIndex = mkEdgeIndex (array2D [| [| 0L; 1L |]; [| 1L; 0L |] |])

    let out = conv.forward (x, edgeIndex)
    out.shape |> should equal [| 3L; 8L |]
    conv.Bias |> should equal None

[<Fact>]
let ``SAGEConv output has gradients`` () =
    let conv = SAGEConv.init 4 8 torch.float32 torch.CPU

    let x = torch.randn ([| 3L; 4L |], dtype = torch.float32, device = torch.CPU)

    let edgeIndex =
        mkEdgeIndex (array2D [| [| 0L; 1L; 1L; 2L |]; [| 1L; 0L; 2L; 1L |] |])

    let out = conv.forward (x, edgeIndex)
    let loss = out.sum ()
    loss.backward ()
    conv.WeightSelf.requires_grad |> should equal true
    conv.WeightNeighbor.requires_grad |> should equal true

// --- Batch tests ---

[<Fact>]
let ``Batch combines two graphs`` () =
    let x1 = torch.randn ([| 3L; 4L |], dtype = torch.float32, device = torch.CPU)
    let ei1 = mkEdgeIndex (array2D [| [| 0L; 1L |]; [| 1L; 2L |] |])
    let g1 = GraphData.create x1 ei1

    let x2 = torch.randn ([| 2L; 4L |], dtype = torch.float32, device = torch.CPU)
    let ei2 = mkEdgeIndex (array2D [| [| 0L |]; [| 1L |] |])
    let g2 = GraphData.create x2 ei2

    let batched = Batch.batch [| g1; g2 |]
    GraphData.numNodes batched |> should equal 5L
    GraphData.numEdges batched |> should equal 3L
    Batch.numGraphs batched |> should equal 2L

    let bv = batched.Batch |> Option.get
    bv.shape |> should equal [| 5L |]
    // First 3 nodes belong to graph 0, next 2 to graph 1
    bv[0] |> scalarF32 |> should (equalWithin 1e-5) 0.0f
    bv[2] |> scalarF32 |> should (equalWithin 1e-5) 0.0f
    bv[3] |> scalarF32 |> should (equalWithin 1e-5) 1.0f
    bv[4] |> scalarF32 |> should (equalWithin 1e-5) 1.0f

// --- GlobalPool tests ---

[<Fact>]
let ``globalMeanPool computes per-graph mean`` () =
    let x =
        torch.tensor (array2D [| [| 2.0f; 4.0f |]; [| 4.0f; 6.0f |]; [| 1.0f; 3.0f |] |], device = torch.CPU)


    let batch = (TorchSharp.torch.tensor ([| 0L; 0L; 1L |]: int64 array))


    let out = GlobalPool.globalMeanPool x batch 2L
    out.shape |> should equal [| 2L; 2L |]
    // Graph 0: mean([2,4],[4,6]) = [3,5]
    out.at [ I 0; I 0 ]
    |> scalarF32
    |> should (equalWithin 1e-5) 3.0f

    out.at [ I 0; I 1 ]
    |> scalarF32
    |> should (equalWithin 1e-5) 5.0f
    // Graph 1: [1,3]
    out.at [ I 1; I 0 ]
    |> scalarF32
    |> should (equalWithin 1e-5) 1.0f

    out.at [ I 1; I 1 ]
    |> scalarF32
    |> should (equalWithin 1e-5) 3.0f

[<Fact>]
let ``globalSumPool computes per-graph sum`` () =
    let x =
        torch.tensor (array2D [| [| 1.0f; 2.0f |]; [| 3.0f; 4.0f |]; [| 5.0f; 6.0f |] |], device = torch.CPU)


    let batch = (TorchSharp.torch.tensor ([| 0L; 0L; 1L |]: int64 array))


    let out = GlobalPool.globalSumPool x batch 2L
    out.shape |> should equal [| 2L; 2L |]
    // Graph 0: sum([1,2],[3,4]) = [4,6]
    out.at [ I 0; I 0 ]
    |> scalarF32
    |> should (equalWithin 1e-5) 4.0f

    out.at [ I 0; I 1 ]
    |> scalarF32
    |> should (equalWithin 1e-5) 6.0f
    // Graph 1: [5,6]
    out.at [ I 1; I 0 ]
    |> scalarF32
    |> should (equalWithin 1e-5) 5.0f

    out.at [ I 1; I 1 ]
    |> scalarF32
    |> should (equalWithin 1e-5) 6.0f

[<Fact>]
let ``globalMaxPool computes per-graph max`` () =
    let x =
        torch.tensor (array2D [| [| 1.0f; 4.0f |]; [| 3.0f; 2.0f |]; [| 5.0f; 6.0f |] |], device = torch.CPU)


    let batch = (TorchSharp.torch.tensor ([| 0L; 0L; 1L |]: int64 array))


    let out = GlobalPool.globalMaxPool x batch 2L
    out.shape |> should equal [| 2L; 2L |]
    // Graph 0: max([1,4],[3,2]) = [3,4]
    out.at [ I 0; I 0 ]
    |> scalarF32
    |> should (equalWithin 1e-5) 3.0f

    out.at [ I 0; I 1 ]
    |> scalarF32
    |> should (equalWithin 1e-5) 4.0f
    // Graph 1: [5,6]
    out.at [ I 1; I 0 ]
    |> scalarF32
    |> should (equalWithin 1e-5) 5.0f

    out.at [ I 1; I 1 ]
    |> scalarF32
    |> should (equalWithin 1e-5) 6.0f

// --- GINConv tests ---

[<Fact>]
let ``GINConv forward produces correct shape`` () =
    let conv = GINConv.init 4 16 8 false torch.float32 torch.CPU

    let x = torch.randn ([| 3L; 4L |], dtype = torch.float32, device = torch.CPU)

    let edgeIndex =
        mkEdgeIndex (array2D [| [| 0L; 1L; 1L; 2L |]; [| 1L; 0L; 2L; 1L |] |])

    let out = conv.forward (x, edgeIndex)
    out.shape |> should equal [| 3L; 8L |]

[<Fact>]
let ``GINConv with trainable eps has gradient`` () =
    let conv = GINConv.init 4 16 8 true torch.float32 torch.CPU

    let x = torch.randn ([| 3L; 4L |], dtype = torch.float32, device = torch.CPU)

    let edgeIndex =
        mkEdgeIndex (array2D [| [| 0L; 1L; 1L; 2L |]; [| 1L; 0L; 2L; 1L |] |])

    let out = conv.forward (x, edgeIndex)
    let loss = out.sum ()
    loss.backward ()
    conv.Eps.requires_grad |> should equal true
    conv.Linear1.Weight.requires_grad |> should equal true

[<Fact>]
let ``GINConv non-trainable eps has no gradient`` () =
    let conv = GINConv.init 4 16 8 false torch.float32 torch.CPU
    conv.Eps.requires_grad |> should equal false

// --- GraphNorm tests ---

[<Fact>]
let ``GraphNorm forward produces correct shape`` () =
    let norm = GraphNorm.init 4 torch.float32 torch.CPU
    let x = torch.randn ([| 5L; 4L |], dtype = torch.float32, device = torch.CPU)

    let batch = (TorchSharp.torch.tensor ([| 0L; 0L; 0L; 1L; 1L |]: int64 array))


    let out = norm.forward (x, Some batch)
    out.shape |> should equal [| 5L; 4L |]

[<Fact>]
let ``GraphNorm without batch treats all nodes as one graph`` () =
    let norm = GraphNorm.init 4 torch.float32 torch.CPU
    let x = torch.randn ([| 5L; 4L |], dtype = torch.float32, device = torch.CPU)
    let out = norm.forward (x, None)
    out.shape |> should equal [| 5L; 4L |]

[<Fact>]
let ``GraphNorm per-graph mean is near zero`` () =
    let norm = GraphNorm.init 4 torch.float32 torch.CPU
    let x = torch.randn ([| 6L; 4L |], dtype = torch.float32, device = torch.CPU)

    let batch = (TorchSharp.torch.tensor ([| 0L; 0L; 0L; 1L; 1L; 1L |]: int64 array))


    let out = norm.forward (x, Some batch)
    // Per-graph mean should be near 0 after normalization (with default alpha=1, gamma=1, beta=0)
    let graphMean = GlobalPool.globalMeanPool out batch 2L
    let meanAbs = graphMean.abs().sum().item<float32> ()
    meanAbs |> should be (lessThan 0.1f)
