namespace Toro.GNN

open TorchSharp
open Toro

/// Aggregation strategy for message passing.
type Aggregation =
    | Add
    | Mean
    | Max

/// Shared message-passing utilities used by all graph convolution layers.
module MessagePassing =

    /// Scatter messages [E, F] to target nodes [N, F] using the given aggregation.
    let aggregate (aggr: Aggregation) (msg: Tensor) (targetIdx: Tensor) (numNodes: int64) (features: int64) : Tensor =
        let idx = targetIdx.at([ A; N ]).expand [| msg.shape[0]; features |]

        match aggr with
        | Add ->
            let out =
                torch.zeros ([| numNodes; features |], dtype = msg.dtype, device = msg.device)

            out.scatter_add (0L, idx, msg)
        | Mean ->
            let sum =
                torch.zeros ([| numNodes; features |], dtype = msg.dtype, device = msg.device)

            let sum = sum.scatter_add (0L, idx, msg)

            let count = torch.zeros ([| numNodes |], dtype = msg.dtype, device = msg.device)

            let ones = torch.ones ([| msg.shape[0] |], dtype = msg.dtype, device = msg.device)
            let count = count.scatter_add (0L, targetIdx, ones)
            let count = count.clamp (min = scalar 1.0, max = scalar 1e30)
            sum / count.at [ A; N ]
        | Max ->
            torch.stack (
                [|
                    for i in 0L .. numNodes - 1L do
                        let mask = targetIdx.eq (scalar (float i))
                        let indices = mask.nonzero().squeeze 1L

                        if indices.NumberOfElements > 0L then
                            let selected = msg.index_select (0L, indices)
                            let struct (maxVals, _) = selected.max (0L)
                            yield maxVals
                        else
                            yield torch.zeros ([| features |], dtype = msg.dtype, device = msg.device)
                |],
                0L
            )

    /// Edge-wise softmax: softmax of scores grouped by target node index.
    /// scores: [E] or [E, H], targetIdx: [E], numNodes: N.
    /// Returns normalized attention coefficients with the same shape as scores.
    let edgeSoftmax (scores: Tensor) (targetIdx: Tensor) (numNodes: int64) : Tensor =
        let expScores = scores.exp ()

        if int expScores.ndim = 1 then
            let denom =
                torch.zeros ([| numNodes |], dtype = expScores.dtype, device = expScores.device)

            let denom = denom.scatter_add (0L, targetIdx, expScores)
            expScores / denom[targetIdx]
        else
            let heads = expScores.shape[1]
            let idx = targetIdx.at([ A; N ]).expand [| expScores.shape[0]; heads |]

            let denom =
                torch.zeros ([| numNodes; heads |], dtype = expScores.dtype, device = expScores.device)

            let denom = denom.scatter_add (0L, idx, expScores)
            expScores / denom.at [ T targetIdx; A ]
