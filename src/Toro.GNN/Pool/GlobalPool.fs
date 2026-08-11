namespace Toro.GNN

open Toro

/// Global graph pooling operations.
/// Aggregate node features into a single graph-level representation.
module GlobalPool =
    /// Mean of node features per graph. batch: [N] maps nodes to graph index.
    let globalMeanPool (x: Tensor) (batch: Tensor) (numGraphs: int) : Tensor =
        let features = x.Shape[1]

        let sum = Tensor.zeros ([ numGraphs; features ], x.DType, x.Device)
        let batchIdx = batch.unsqueeze 1
        let batchIdx = batchIdx.expand [ x.Shape[0]; features ]
        let sum = sum.scatterAdd (0, batchIdx, x)

        let count = Tensor.zeros ([ numGraphs ], x.DType, x.Device)
        let ones = Tensor.ones ([ x.Shape[0] ], x.DType, x.Device)
        let count = count.scatterAdd (0, batch, ones)
        let count = count.unsqueeze 1
        sum.div count

    /// Sum of node features per graph.
    let globalSumPool (x: Tensor) (batch: Tensor) (numGraphs: int) : Tensor =
        let features = x.Shape[1]

        let out = Tensor.zeros ([ numGraphs; features ], x.DType, x.Device)
        let batchIdx = batch.unsqueeze 1
        let batchIdx = batchIdx.expand [ x.Shape[0]; features ]
        out.scatterAdd (0, batchIdx, x)

    /// Max of node features per graph.
    let globalMaxPool (x: Tensor) (batch: Tensor) (numGraphs: int) : Tensor =
        let results = ResizeArray<Tensor>()

        for i in 0 .. numGraphs - 1 do
            let mask = batch.eqScalar (float i)
            let indices = mask.nonzero().squeeze 1
            let selected = x.indexSelect (0, indices)
            let maxVals, _ = selected.max 0
            results.Add(maxVals)

        Tensor.stack (results |> Seq.toList, 0)
