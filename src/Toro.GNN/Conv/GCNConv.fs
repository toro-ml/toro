namespace Toro.GNN

open Toro
open Toro.NN

/// Graph Convolutional Network layer (Kipf & Welling, 2017).
/// $\mathbf{x}_i' = \sum_{j \in \mathcal{N}(i) \cup \{i\}}
///   \frac{1}{\sqrt{\deg(i)} \cdot \sqrt{\deg(j)}}
///   \left( \mathbf{W}^\top \mathbf{x}_j \right) + \mathbf{b}$
type GCNConv = {
    Weight: Tensor
    Bias: Tensor option
} with

    member this.forward(x: Tensor, edgeIndex: Tensor) : Result<Tensor, ToroError> =
        let numNodes = x.Shape[0]
        let outChannels = this.Weight.Shape[0]

        result {
            let! edgeIndex = GraphUtils.addSelfLoops edgeIndex numNodes

            // Linear transform
            let! wt = this.Weight.t ()
            let! h = x.matmul wt

            // Source / target indices from edge_index [2, E]
            let src = edgeIndex[0]
            let tgt = edgeIndex[1]

            // Symmetric normalization: D^{-1/2} A D^{-1/2}
            let! deg = GraphUtils.degree tgt numNodes h.DType h.Device
            let! degInvSqrt = deg.pow -0.5
            let! zero = Tensor.zeros ([ numNodes ], h.DType, h.Device)
            let! degInvSqrt = Tensor.where (degInvSqrt.eqScalar infinity, zero, degInvSqrt)

            // Per-edge normalization coefficient
            let norm = degInvSqrt[src] * degInvSqrt[tgt]

            // Gather and normalize messages, then aggregate
            let msg = h[src] * norm.at [ A; N ]
            let! out = MessagePassing.aggregate Add msg tgt numNodes outChannels

            match this.Bias with
            | None -> return out
            | Some bias -> return! out.add bias
        }

module GCNConv =
    /// Create a GCNConv layer with bias.
    let init (inChannels: int) (outChannels: int) (dtype: DType) (device: Device) : Result<GCNConv, ToroError> =
        result {
            let! w = Init.toParam [ outChannels; inChannels ] dtype device Init.defaultKaimingNormal
            let bound = 1.0 / sqrt (float outChannels)
            let! b = Init.toParam [ outChannels ] dtype device (Init.Uniform(-bound, bound))
            return { Weight = w; Bias = Some b }
        }

    /// Create a GCNConv layer without bias.
    let initNoBias (inChannels: int) (outChannels: int) (dtype: DType) (device: Device) : Result<GCNConv, ToroError> =
        result {
            let! w = Init.toParam [ outChannels; inChannels ] dtype device Init.defaultKaimingNormal
            return { Weight = w; Bias = None }
        }
