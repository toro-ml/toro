namespace Toro.GNN

open Toro
open Toro.NN

/// Graph Attention Network layer (Velickovic et al., 2018).
/// Multi-head attention mechanism over graph edges.
/// $\alpha_{ij} = \text{softmax}_j(\text{LeakyReLU}(\mathbf{a}^T [\mathbf{W}\mathbf{x}_i \| \mathbf{W}\mathbf{x}_j]))$
type GATConv = {
    Weight: Tensor
    AttSrc: Tensor
    AttTgt: Tensor
    Bias: Tensor option
    Heads: int
    OutChannels: int
    NegativeSlope: float
    Concat: bool
} with

    member this.forward(x: Tensor, edgeIndex: Tensor) : Result<Tensor, ToroError> =
        let numNodes = x.Shape[0]

        result {
            // Linear transform: [N, inChannels] -> [N, heads * outChannels]
            let! wt = this.Weight.t ()
            let! h = x.matmul wt
            // Reshape to [N, heads, outChannels]
            let! h = h.reshape [ numNodes; this.Heads; this.OutChannels ]

            // Attention scores per node: (h * att).sum(-1) -> [N, heads]
            let! alphaSrc = h.mul this.AttSrc
            let! alphaSrc = alphaSrc.sum (2, keepDim = false)
            let! alphaTgt = h.mul this.AttTgt
            let! alphaTgt = alphaTgt.sum (2, keepDim = false)

            // Source / target indices
            let src = edgeIndex[0]
            let tgt = edgeIndex[1]

            // Per-edge attention: alphaSrc[src] + alphaTgt[tgt] -> [E, heads]
            let edgeAlpha = alphaSrc.at [ T src; A ] + alphaTgt.at [ T tgt; A ]

            // LeakyReLU + edge-softmax
            let! edgeAlpha = edgeAlpha.leakyRelu this.NegativeSlope
            let! attn = MessagePassing.edgeSoftmax edgeAlpha tgt numNodes

            // Weighted messages: h[src] * alpha -> [E, heads, outChannels]
            let! attnExp = attn.unsqueeze 2
            let msgSrc = h.at [ T src; A; A ]
            let! msg = msgSrc.mul attnExp

            // Aggregate (scatter add) -> [N, heads, outChannels]
            let numEdges = msg.Shape[0]
            let! msg = msg.reshape [ numEdges; this.Heads * this.OutChannels ]
            let! out = MessagePassing.aggregate Add msg tgt numNodes (this.Heads * this.OutChannels)
            let! out = out.reshape [ numNodes; this.Heads; this.OutChannels ]

            // Concat or mean over heads
            let! out =
                if this.Concat then
                    out.reshape [ numNodes; this.Heads * this.OutChannels ]
                else
                    out.mean (1, keepDim = false)

            match this.Bias with
            | None -> return out
            | Some bias -> return! out.add bias
        }

module GATConv =
    /// Create a GATConv layer.
    /// concat=true: output dim = heads * outChannels.
    /// concat=false: output dim = outChannels (mean over heads).
    let init
        (inChannels: int)
        (outChannels: int)
        (heads: int)
        (concat: bool)
        (negativeSlope: float)
        (dtype: DType)
        (device: Device)
        : Result<GATConv, ToroError> =
        result {
            let! w = Init.toParam [ heads * outChannels; inChannels ] dtype device Init.defaultKaimingNormal
            let! attSrc = Init.toParam [ 1; heads; outChannels ] dtype device (Init.Uniform(-1.0, 1.0))
            let! attTgt = Init.toParam [ 1; heads; outChannels ] dtype device (Init.Uniform(-1.0, 1.0))

            let totalOut = if concat then heads * outChannels else outChannels
            let bound = 1.0 / sqrt (float totalOut)
            let! b = Init.toParam [ totalOut ] dtype device (Init.Uniform(-bound, bound))

            return {
                Weight = w
                AttSrc = attSrc
                AttTgt = attTgt
                Bias = Some b
                Heads = heads
                OutChannels = outChannels
                NegativeSlope = negativeSlope
                Concat = concat
            }
        }

    /// Create a GATConv layer with default settings (heads=1, concat=true, slope=0.2).
    let initDefault (inChannels: int) (outChannels: int) (dtype: DType) (device: Device) : Result<GATConv, ToroError> =
        init inChannels outChannels 1 true 0.2 dtype device
