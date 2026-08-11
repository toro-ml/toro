namespace Toro.NN

open Toro

type MultiHeadAttention = {
    WQ: Linear
    WK: Linear
    WV: Linear
    WO: Linear
    NumHeads: int
    HeadDim: int
} with

    member this.forward(x: Tensor, ?mask: Tensor, ?kvCache: KvCache) : Tensor =
        let batchSz = x.Shape[0]
        let seqLen = x.Shape[1]

        let q = this.WQ.forward x
        let k = this.WK.forward x
        let v = this.WV.forward x

        let q = q.reshape [ batchSz; seqLen; this.NumHeads; this.HeadDim ]
        let q = q.permute [ 0; 2; 1; 3 ]
        let k = k.reshape [ batchSz; seqLen; this.NumHeads; this.HeadDim ]
        let k = k.permute [ 0; 2; 1; 3 ]
        let v = v.reshape [ batchSz; seqLen; this.NumHeads; this.HeadDim ]
        let v = v.permute [ 0; 2; 1; 3 ]

        let k, v =
            match kvCache with
            | Some cache -> cache.append (k, v)
            | None -> k, v

        let attn = q.scaledDotProductAttention (k, v, ?attnMask = mask)
        let attn = attn.permute [ 0; 2; 1; 3 ]
        let attn = attn.contiguous ()
        let attn = attn.reshape [ batchSz; seqLen; this.NumHeads * this.HeadDim ]
        this.WO.forward attn

module MultiHeadAttention =
    let init (dim: int) (numHeads: int) (dtype: DType) (device: Device) : MultiHeadAttention =
        let headDim = dim / numHeads

        let wq = Linear.initNoBias dim dim dtype device
        let wk = Linear.initNoBias dim dim dtype device
        let wv = Linear.initNoBias dim dim dtype device
        let wo = Linear.initNoBias dim dim dtype device

        {
            WQ = wq
            WK = wk
            WV = wv
            WO = wo
            NumHeads = numHeads
            HeadDim = headDim
        }

type TransformerBlock = {
    Attn: MultiHeadAttention
    Norm1: LayerNorm
    Norm2: LayerNorm
    Ff1: Linear
    Ff2: Linear
} with

    member this.forward(x: Tensor, ?mask: Tensor) : Tensor =
        let normed = this.Norm1.forward x
        let attnOut = this.Attn.forward (normed, ?mask = mask)
        let x = x.add attnOut

        let normed = this.Norm2.forward x
        let h = this.Ff1.forward normed
        let h = h.gelu ()
        let ffOut = this.Ff2.forward h
        x.add ffOut

module TransformerBlock =
    let init (dim: int) (numHeads: int) (ffDim: int) (dtype: DType) (device: Device) : TransformerBlock =
        let attn = MultiHeadAttention.init dim numHeads dtype device
        let norm1 = LayerNorm.initDefault dim dtype device
        let norm2 = LayerNorm.initDefault dim dtype device
        let ff1 = Linear.init dim ffDim dtype device
        let ff2 = Linear.init ffDim dim dtype device

        {
            Attn = attn
            Norm1 = norm1
            Norm2 = norm2
            Ff1 = ff1
            Ff2 = ff2
        }
