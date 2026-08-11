namespace Toro.GNN

open Toro
open Toro.NN

/// GraphSAGE convolution layer (Hamilton et al., 2017).
/// $\mathbf{x}_i' = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W}_2 \cdot \text{mean}_{j \in \mathcal{N}(i)} \mathbf{x}_j$
type SAGEConv = {
    WeightSelf: Tensor
    WeightNeighbor: Tensor
    Bias: Tensor option
} with

    member this.forward(x: Tensor, edgeIndex: Tensor) : Tensor =
        let numNodes = x.Shape[0]
        let inChannels = x.Shape[1]

        let src = edgeIndex[0]
        let tgt = edgeIndex[1]

        // Aggregate neighbor features with mean
        let neighborMsg = x.at [ T src; A ]
        let aggr = MessagePassing.aggregate Mean neighborMsg tgt numNodes inChannels

        // Self transform: x @ W_self^T
        let wSelfT = this.WeightSelf.t ()
        let selfOut = x.matmul wSelfT

        // Neighbor transform: aggr @ W_neighbor^T
        let wNeighborT = this.WeightNeighbor.t ()
        let neighborOut = aggr.matmul wNeighborT

        // Combine
        let out = selfOut.add neighborOut

        match this.Bias with
        | None -> out
        | Some bias -> out.add bias

module SAGEConv =
    /// Create a SAGEConv layer with bias.
    let init (inChannels: int) (outChannels: int) (dtype: DType) (device: Device) : SAGEConv =
        let wSelf =
            Init.toParam [ outChannels; inChannels ] dtype device Init.defaultKaimingNormal

        let wNeighbor =
            Init.toParam [ outChannels; inChannels ] dtype device Init.defaultKaimingNormal

        let bound = 1.0 / sqrt (float outChannels)
        let b = Init.toParam [ outChannels ] dtype device (Init.Uniform(-bound, bound))

        {
            WeightSelf = wSelf
            WeightNeighbor = wNeighbor
            Bias = Some b
        }

    /// Create a SAGEConv layer without bias.
    let initNoBias (inChannels: int) (outChannels: int) (dtype: DType) (device: Device) : SAGEConv =
        let wSelf =
            Init.toParam [ outChannels; inChannels ] dtype device Init.defaultKaimingNormal

        let wNeighbor =
            Init.toParam [ outChannels; inChannels ] dtype device Init.defaultKaimingNormal

        {
            WeightSelf = wSelf
            WeightNeighbor = wNeighbor
            Bias = None
        }
