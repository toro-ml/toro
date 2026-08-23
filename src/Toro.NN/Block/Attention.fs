namespace Toro.NN

open TorchSharp
open Toro

/// Helpers for scaled-dot-product attention masks.
module Attention =

    /// Convert a padding mask `[batch, seq]` (1 = attend, 0 = pad) into an additive
    /// SDPA mask `[batch, 1, 1, seq]`. Padded positions receive a dtype-aware
    /// negative saturation value (`-1e4` for Float16/BFloat16, `-1e9` otherwise).
    let additiveMask (paddingMask: Tensor) (dtype: torch.ScalarType) : Tensor =
        let mask = paddingMask.to_type(dtype).unsqueeze(1L).unsqueeze 2L

        let negInf =
            match dtype with
            | torch.ScalarType.Float16
            | torch.ScalarType.BFloat16 -> -1.0e4
            | _ -> -1.0e9

        (scalar 1.0 - mask) * scalar negInf

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

        let toHeads (projection: Linear) =
            projection.forward x
            |> _.reshape([| batchSz; seqLen; this.NumHeads; this.HeadDim |])
            |> _.permute([| 0L; 2L; 1L; 3L |])

        let q = toHeads this.WQ
        let k = toHeads this.WK
        let v = toHeads this.WV

        let k, v =
            match kvCache with
            | Some cache -> cache.append (k, v)
            | None -> k, v

        let attended =
            torch.nn.functional.scaled_dot_product_attention (q, k, v, attn_mask = (mask |> Option.defaultValue null))

        attended.permute([| 0L; 2L; 1L; 3L |]).contiguous().reshape [| batchSz; seqLen; this.NumHeads * this.HeadDim |]
        |> this.WO.forward

module MultiHeadAttention =

    let private create (linearInit: int64 -> int64 -> torch.ScalarType -> torch.Device -> Linear) dim numHeads dtype device =
        let headDim = dim / numHeads

        {
            WQ = linearInit dim dim dtype device
            WK = linearInit dim dim dtype device
            WV = linearInit dim dim dtype device
            WO = linearInit dim dim dtype device
            NumHeads = numHeads
            HeadDim = headDim
        }

    /// Create multi-head attention with bias on the projections.
    let init (dim: int64) (numHeads: int64) (dtype: torch.ScalarType) (device: torch.Device) : MultiHeadAttention =
        create Linear.init dim numHeads dtype device

    /// Create multi-head attention without bias on the projections.
    let initNoBias (dim: int64) (numHeads: int64) (dtype: torch.ScalarType) (device: torch.Device) : MultiHeadAttention =
        create Linear.initNoBias dim numHeads dtype device

/// Pre-norm transformer block: Norm → Attn → Add → Norm → FFN → Add.
type PreNormTransformerBlock = {
    Attn: MultiHeadAttention
    AttnNorm: LayerNorm
    Ff1: Linear
    Ff2: Linear
    FfNorm: LayerNorm
} with

    member this.forward(x: Tensor, ?mask: Tensor) : Tensor =
        let attnOut = this.Attn.forward (this.AttnNorm.forward x, ?mask = mask)
        let x = x.add attnOut

        let ffOut =
            this.Ff1.forward (this.FfNorm.forward x)
            |> fun h -> h.gelu () |> this.Ff2.forward

        x.add ffOut

module PreNormTransformerBlock =

    /// Create a pre-norm block with unbiased attention and the given LayerNorm config.
    let init
        (dim: int64)
        (numHeads: int64)
        (ffDim: int64)
        (lnConfig: LayerNormConfig)
        (dtype: torch.ScalarType)
        (device: torch.Device)
        : PreNormTransformerBlock =
        {
            Attn = MultiHeadAttention.initNoBias dim numHeads dtype device
            AttnNorm = LayerNorm.init dim lnConfig dtype device
            Ff1 = Linear.init dim ffDim dtype device
            Ff2 = Linear.init ffDim dim dtype device
            FfNorm = LayerNorm.init dim lnConfig dtype device
        }

    /// Create a pre-norm block with the default LayerNorm config.
    let initDefault
        (dim: int64)
        (numHeads: int64)
        (ffDim: int64)
        (dtype: torch.ScalarType)
        (device: torch.Device)
        : PreNormTransformerBlock =
        init dim numHeads ffDim LayerNormConfig.defaultConfig dtype device

/// Post-norm transformer block: Attn → Add → Norm → FFN → Add → Norm.
type PostNormTransformerBlock = {
    Attn: MultiHeadAttention
    AttnNorm: LayerNorm
    Ff1: Linear
    Ff2: Linear
    FfNorm: LayerNorm
} with

    /// $x \leftarrow \mathrm{LN}(x + \mathrm{Attn}(x))$, then $x \leftarrow \mathrm{LN}(x + \mathrm{FFN}(x))$.
    member this.forward(x: Tensor, ?mask: Tensor) : Tensor =
        let attnOut = this.Attn.forward (x, ?mask = mask)
        let x = this.AttnNorm.forward (x.add attnOut)
        let ffOut = this.Ff1.forward x |> fun h -> h.gelu () |> this.Ff2.forward
        this.FfNorm.forward (x.add ffOut)

module PostNormTransformerBlock =

    /// Create a post-norm block with biased attention and the given LayerNorm config.
    let init
        (dim: int64)
        (numHeads: int64)
        (ffDim: int64)
        (lnConfig: LayerNormConfig)
        (dtype: torch.ScalarType)
        (device: torch.Device)
        : PostNormTransformerBlock =
        {
            Attn = MultiHeadAttention.init dim numHeads dtype device
            AttnNorm = LayerNorm.init dim lnConfig dtype device
            Ff1 = Linear.init dim ffDim dtype device
            Ff2 = Linear.init ffDim dim dtype device
            FfNorm = LayerNorm.init dim lnConfig dtype device
        }

    /// Create a post-norm block with the default LayerNorm config.
    let initDefault
        (dim: int64)
        (numHeads: int64)
        (ffDim: int64)
        (dtype: torch.ScalarType)
        (device: torch.Device)
        : PostNormTransformerBlock =
        init dim numHeads ffDim LayerNormConfig.defaultConfig dtype device
