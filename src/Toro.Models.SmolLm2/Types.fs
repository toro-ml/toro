namespace Toro.Models

open System
open Toro.NN

/// Configuration values used by a SmolLM2 causal language model.
type SmolLm2Config = {
    VocabSize: int64
    HiddenSize: int64
    IntermediateSize: int64
    NumHiddenLayers: int
    NumAttentionHeads: int64
    NumKeyValueHeads: int64
    MaxPositionEmbeddings: int64
    RmsNormEps: float
    RopeTheta: float
    BosTokenId: int64
    EosTokenId: int64
}

/// Validation operations for SmolLM2 configurations.
module SmolLm2Config =

    /// Validate dimensions and constants required by the implemented architecture.
    let validate (config: SmolLm2Config) =
        if
            config.VocabSize <= 0L
            || config.HiddenSize <= 0L
            || config.IntermediateSize <= 0L
            || config.NumHiddenLayers <= 0
            || config.NumAttentionHeads <= 0L
            || config.NumKeyValueHeads <= 0L
            || config.MaxPositionEmbeddings <= 0L
            || config.RmsNormEps <= 0.0
            || not (Double.IsFinite config.RmsNormEps)
            || config.RopeTheta <= 0.0
            || not (Double.IsFinite config.RopeTheta)
        then
            invalidArg (nameof config) "SmolLM2 dimensions and numeric constants must be finite and positive."

        if config.HiddenSize % config.NumAttentionHeads <> 0L then
            invalidArg (nameof config) "SmolLM2 hidden size must be divisible by the number of attention heads."

        let headSize = config.HiddenSize / config.NumAttentionHeads

        if headSize % 2L <> 0L then
            invalidArg (nameof config) "SmolLM2 attention head size must be even for rotary embedding."

        if config.NumAttentionHeads % config.NumKeyValueHeads <> 0L then
            invalidArg (nameof config) "SmolLM2 attention heads must be divisible by key/value heads."

        if
            config.BosTokenId < 0L
            || config.BosTokenId >= config.VocabSize
        then
            invalidArg (nameof config) "SmolLM2 BOS token ID must be within the vocabulary."

        if
            config.EosTokenId < 0L
            || config.EosTokenId >= config.VocabSize
        then
            invalidArg (nameof config) "SmolLM2 EOS token ID must be within the vocabulary."

/// Projection layers in one SmolLM2 grouped-query attention block.
type SmolLm2Attention = {
    Query: Linear
    Key: Linear
    Value: Linear
    Output: Linear
}

/// Projection layers in one SmolLM2 SwiGLU feed-forward block.
type SmolLm2Mlp = {
    Gate: Linear
    Up: Linear
    Down: Linear
}

/// One SmolLM2 transformer block.
type SmolLm2Block = {
    InputNorm: RmsNorm
    Attention: SmolLm2Attention
    PostAttentionNorm: RmsNorm
    Mlp: SmolLm2Mlp
}
