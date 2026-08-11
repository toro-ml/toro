namespace Toro.GNN

open TorchSharp
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
    Heads: int64
    OutChannels: int64
    NegativeSlope: float
    Concat: bool
} with

    member this.forward(x: Tensor, edgeIndex: Tensor) : Tensor =
        let numNodes = x.shape[0]

        // Linear transform: [N, inChannels] -> [N, heads * outChannels]
        let wt = this.Weight.t ()
        let h = x.matmul wt
        // Reshape to [N, heads, outChannels]
        let h = h.reshape [| numNodes; this.Heads; this.OutChannels |]

        // Attention scores per node: (h * att).sum(-1) -> [N, heads]
        let alphaSrc = h.mul this.AttSrc
        let alphaSrc = alphaSrc.sum [| 2L |]
        let alphaTgt = h.mul this.AttTgt
        let alphaTgt = alphaTgt.sum [| 2L |]

        // Source / target indices
        let src = edgeIndex[0]
        let tgt = edgeIndex[1]

        // Per-edge attention: alphaSrc[src] + alphaTgt[tgt] -> [E, heads]
        let edgeAlpha = alphaSrc.at [ T src; A ] + alphaTgt.at [ T tgt; A ]

        // LeakyReLU + edge-softmax
        let edgeAlpha = torch.nn.functional.leaky_relu (edgeAlpha, this.NegativeSlope)
        let attn = MessagePassing.edgeSoftmax edgeAlpha tgt numNodes

        // Weighted messages: h[src] * alpha -> [E, heads, outChannels]
        let attnExp = attn.unsqueeze 2L
        let msgSrc = h.at [ T src; A; A ]
        let msg = msgSrc.mul attnExp

        // Aggregate (scatter add) -> [N, heads, outChannels]
        let numEdges = msg.shape[0]
        let msg = msg.reshape [| numEdges; this.Heads * this.OutChannels |]

        let out =
            MessagePassing.aggregate Add msg tgt numNodes (this.Heads * this.OutChannels)

        let out = out.reshape [| numNodes; this.Heads; this.OutChannels |]

        // Concat or mean over heads
        let out =
            if this.Concat then
                out.reshape [| numNodes; this.Heads * this.OutChannels |]
            else
                out.mean [| 1L |]

        match this.Bias with
        | None -> out
        | Some bias -> out.add bias

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
        (dtype: torch.ScalarType)
        (device: torch.Device)
        : GATConv =
        let w =
            Init.toParam [| int64 (heads * outChannels); int64 inChannels |] dtype device Init.defaultKaimingNormal

        let attSrc =
            Init.toParam [| 1L; int64 heads; int64 outChannels |] dtype device (Init.Uniform(-1.0, 1.0))

        let attTgt =
            Init.toParam [| 1L; int64 heads; int64 outChannels |] dtype device (Init.Uniform(-1.0, 1.0))

        let totalOut = if concat then heads * outChannels else outChannels
        let bound = 1.0 / sqrt (float totalOut)
        let b = Init.toParam [| int64 totalOut |] dtype device (Init.Uniform(-bound, bound))

        {
            Weight = w
            AttSrc = attSrc
            AttTgt = attTgt
            Bias = Some b
            Heads = heads
            OutChannels = outChannels
            NegativeSlope = negativeSlope
            Concat = concat
        }

    /// Create a GATConv layer with default settings (heads=1, concat=true, slope=0.2).
    let initDefault (inChannels: int) (outChannels: int) (dtype: torch.ScalarType) (device: torch.Device) : GATConv =
        init inChannels outChannels 1 true 0.2 dtype device
