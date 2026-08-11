namespace Toro.NN

open TorchSharp
open Toro

type MultiHeadAttention = {
    WQ: Linear
    WK: Linear
    WV: Linear
    WO: Linear
    NumHeads: int64
    HeadDim: int64
} with

    member this.forward(x: Tensor, ?mask: Tensor, ?kvCache: KvCache) : Tensor =
        let batchSz = x.shape[0]
        let seqLen = x.shape[1]

        let q = this.WQ.forward x
        let k = this.WK.forward x
        let v = this.WV.forward x

        let q = q.reshape [| batchSz; seqLen; this.NumHeads; this.HeadDim |]
        let q = q.permute [| 0L; 2L; 1L; 3L |]

        let k = k.reshape [| batchSz; seqLen; this.NumHeads; this.HeadDim |]
        let k = k.permute [| 0L; 2L; 1L; 3L |]

        let v = v.reshape [| batchSz; seqLen; this.NumHeads; this.HeadDim |]
        let v = v.permute [| 0L; 2L; 1L; 3L |]

        let k, v =
            match kvCache with
            | Some cache -> cache.append (k, v)
            | None -> k, v

        let maskTensor = mask |> Option.defaultValue null

        let attn =
            torch.nn.functional.scaled_dot_product_attention (q, k, v, attn_mask = maskTensor)

        let attn = attn.permute [| 0L; 2L; 1L; 3L |]
        let attn = attn.contiguous ()

        let attn = attn.reshape [| batchSz; seqLen; this.NumHeads * this.HeadDim |]

        this.WO.forward attn

module MultiHeadAttention =
    let init (dim: int64) (numHeads: int64) (dtype: torch.ScalarType) (device: torch.Device) : MultiHeadAttention =
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
    let init (dim: int64) (numHeads: int64) (ffDim: int64) (dtype: torch.ScalarType) (device: torch.Device) : TransformerBlock =
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
