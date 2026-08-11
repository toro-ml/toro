namespace Toro.GNN

open TorchSharp
open Toro

/// Batching multiple graphs into a single disconnected graph (PyG-style).
module Batch =
    /// Combine a list of graphs into one batched graph.
    /// Node features are concatenated along dim 0.
    /// Edge indices are offset by cumulative node counts.
    /// A batch vector [N_total] maps each node to its source graph index.
    let batch (graphs: GraphData array) : GraphData =
        let device = graphs[0].X.device
        let numGraphs = graphs.Length

        let mutable nodeOffset = 0L
        let mutable xs = []
        let mutable edgeIndices = []
        let mutable batchVecs = []
        let mutable edgeAttrs = []
        let mutable hasEdgeAttr = graphs[0].EdgeAttr.IsSome

        for i in 0 .. numGraphs - 1 do
            let g = graphs[i]
            let numNodes = GraphData.numNodes g

            xs <- g.X :: xs

            let offset =
                torch.full ([| 2L; GraphData.numEdges g |], scalar (float nodeOffset), dtype = torch.int64, device = device)

            let offsetEdgeIndex = g.EdgeIndex.add offset
            edgeIndices <- offsetEdgeIndex :: edgeIndices

            let bv =
                torch.full ([| numNodes |], scalar (float i), dtype = torch.int64, device = device)

            batchVecs <- bv :: batchVecs

            if hasEdgeAttr then
                match g.EdgeAttr with
                | Some ea -> edgeAttrs <- ea :: edgeAttrs
                | None -> hasEdgeAttr <- false

            nodeOffset <- nodeOffset + numNodes

        let x = torch.cat (List.rev xs |> List.toArray, 0L)
        let edgeIndex = torch.cat (List.rev edgeIndices |> List.toArray, 1L)
        let batchVec = torch.cat (List.rev batchVecs |> List.toArray, 0L)

        let edgeAttr =
            if hasEdgeAttr && edgeAttrs.Length > 0 then
                Some(torch.cat (List.rev edgeAttrs |> List.toArray, 0L))
            else
                None

        {
            X = x
            EdgeIndex = edgeIndex
            EdgeAttr = edgeAttr
            Batch = Some batchVec
        }

    /// Return the number of graphs in a batched graph.
    let numGraphs (g: GraphData) : int64 =
        match g.Batch with
        | Some b -> b.max().ToInt64() + 1L
        | None -> 1
