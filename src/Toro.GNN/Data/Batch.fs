namespace Toro.GNN

open Toro

/// Batching multiple graphs into a single disconnected graph (PyG-style).
module Batch =
    /// Combine a list of graphs into one batched graph.
    /// Node features are concatenated along dim 0.
    /// Edge indices are offset by cumulative node counts.
    /// A batch vector [N_total] maps each node to its source graph index.
    let batch (graphs: GraphData list) : Result<GraphData, ToroError> =
        result {
            let device = graphs[0].X.Device
            let dtype = graphs[0].X.DType
            let numGraphs = graphs.Length

            let mutable nodeOffset = 0
            let mutable xs = []
            let mutable edgeIndices = []
            let mutable batchVecs = []
            let mutable edgeAttrs = []
            let mutable hasEdgeAttr = graphs[0].EdgeAttr.IsSome

            for i in 0 .. numGraphs - 1 do
                let g = graphs[i]
                let numNodes = GraphData.numNodes g

                xs <- g.X :: xs

                let! offset = Tensor.full ([ 2; GraphData.numEdges g ], float nodeOffset, I64, device)
                let! offsetEdgeIndex = g.EdgeIndex.add offset
                edgeIndices <- offsetEdgeIndex :: edgeIndices

                let! bv = Tensor.full ([ numNodes ], float i, I64, device)
                batchVecs <- bv :: batchVecs

                if hasEdgeAttr then
                    match g.EdgeAttr with
                    | Some ea -> edgeAttrs <- ea :: edgeAttrs
                    | None -> hasEdgeAttr <- false

                nodeOffset <- nodeOffset + numNodes

            let! x = Tensor.cat (List.rev xs, 0)
            let! edgeIndex = Tensor.cat (List.rev edgeIndices, 1)
            let! batchVec = Tensor.cat (List.rev batchVecs, 0)

            let! edgeAttr =
                if hasEdgeAttr && edgeAttrs.Length > 0 then
                    Tensor.cat (List.rev edgeAttrs, 0) |> Result.map Some
                else
                    Ok None

            return {
                X = x
                EdgeIndex = edgeIndex
                EdgeAttr = edgeAttr
                Batch = Some batchVec
            }
        }

    /// Return the number of graphs in a batched graph.
    let numGraphs (g: GraphData) : int =
        match g.Batch with
        | Some b -> int (b.Inner.max().item<int64> ()) + 1
        | None -> 1
