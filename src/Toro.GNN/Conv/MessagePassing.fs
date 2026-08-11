namespace Toro.GNN

open Toro

/// Aggregation strategy for message passing.
type Aggregation =
    | Add
    | Mean
    | Max

/// Shared message-passing utilities used by all graph convolution layers.
module MessagePassing =

    /// Scatter messages [E, F] to target nodes [N, F] using the given aggregation.
    let aggregate (aggr: Aggregation) (msg: Tensor) (targetIdx: Tensor) (numNodes: int) (features: int) : Tensor =
        let idx = targetIdx.at([ A; N ]).expand [ msg.Shape[0]; features ]

        match aggr with
        | Add ->
            let out = Tensor.zeros ([ numNodes; features ], msg.DType, msg.Device)
            out.scatterAdd (0, idx, msg)
        | Mean ->
            let sum = Tensor.zeros ([ numNodes; features ], msg.DType, msg.Device)
            let sum = sum.scatterAdd (0, idx, msg)
            let count = Tensor.zeros ([ numNodes ], msg.DType, msg.Device)
            let ones = Tensor.ones ([ msg.Shape[0] ], msg.DType, msg.Device)
            let count = count.scatterAdd (0, targetIdx, ones)
            let count = count.clamp (1.0, 1e30)
            sum / count.at [ A; N ]
        | Max ->
            let results = ResizeArray<Tensor>()

            for i in 0 .. numNodes - 1 do
                let mask = targetIdx.eqScalar (float i)
                let indices = mask.nonzero().squeeze 1

                if indices.ElemCount > 0L then
                    let selected = msg.indexSelect (0, indices)
                    let maxVals, _ = selected.max 0
                    results.Add maxVals
                else
                    let zeros = Tensor.zeros ([ features ], msg.DType, msg.Device)
                    results.Add zeros

            Tensor.stack (results |> Seq.toList, 0)

    /// Edge-wise softmax: softmax of scores grouped by target node index.
    /// scores: [E] or [E, H], targetIdx: [E], numNodes: N.
    /// Returns normalized attention coefficients with the same shape as scores.
    let edgeSoftmax (scores: Tensor) (targetIdx: Tensor) (numNodes: int) : Tensor =
        let expScores = scores.exp ()

        if expScores.Rank = 1 then
            let denom = Tensor.zeros ([ numNodes ], expScores.DType, expScores.Device)
            let denom = denom.scatterAdd (0, targetIdx, expScores)
            expScores / denom[targetIdx]
        else
            let heads = expScores.Shape[1]
            let idx = targetIdx.at([ A; N ]).expand [ expScores.Shape[0]; heads ]
            let denom = Tensor.zeros ([ numNodes; heads ], expScores.DType, expScores.Device)
            let denom = denom.scatterAdd (0, idx, expScores)
            expScores / denom.at [ T targetIdx; A ]
