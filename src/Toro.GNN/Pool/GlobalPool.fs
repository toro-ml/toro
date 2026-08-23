namespace Toro.GNN

open TorchSharp
open Toro

/// Global graph pooling operations.
/// Aggregate node features into a single graph-level representation.
module GlobalPool =
    /// Mean of node features per graph. batch: [N] maps nodes to graph index.
    let globalMeanPool (x: Tensor) (batch: Tensor) (numGraphs: int64) : Tensor =
        let features = x.shape[1]

        let sum =
            torch.zeros ([| numGraphs; features |], dtype = x.dtype, device = x.device)

        let batchIdx = batch.unsqueeze 1L
        let batchIdx = batchIdx.expand [| x.shape[0]; features |]
        let sum = sum.scatter_add (0L, batchIdx, x)

        let count = torch.zeros ([| numGraphs |], dtype = x.dtype, device = x.device)
        let ones = torch.ones ([| x.shape[0] |], dtype = x.dtype, device = x.device)
        let count = count.scatter_add (0L, batch, ones)
        let count = count.unsqueeze 1L
        sum.div count

    /// Sum of node features per graph.
    let globalSumPool (x: Tensor) (batch: Tensor) (numGraphs: int64) : Tensor =
        let features = x.shape[1]

        let out =
            torch.zeros ([| numGraphs; features |], dtype = x.dtype, device = x.device)

        let batchIdx = batch.unsqueeze 1L
        let batchIdx = batchIdx.expand [| x.shape[0]; features |]
        out.scatter_add (0L, batchIdx, x)

    /// Max of node features per graph.
    let globalMaxPool (x: Tensor) (batch: Tensor) (numGraphs: int64) : Tensor =
        torch.stack (
            [|
                for i in 0L .. numGraphs - 1L do
                    let mask = batch.eq (scalar (float i))
                    let indices = mask.nonzero().squeeze 1L
                    let selected = x.index_select (0L, indices)
                    let struct (maxVals, _) = selected.max (0L)
                    yield maxVals
            |],
            0L
        )
