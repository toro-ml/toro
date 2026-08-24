namespace Toro.Models

open System
open Toro
open Toro.NN

/// Configuration values used by a DistilGPT-2 causal language model.
type DistilGpt2Config = {
    VocabSize: int64
    EmbeddingSize: int64
    IntermediateSize: int64
    NumHiddenLayers: int
    NumAttentionHeads: int64
    MaxPositionEmbeddings: int64
    LayerNormEps: float
    BosTokenId: int64
    EosTokenId: int64
}

/// Validation operations for DistilGPT-2 configurations.
module DistilGpt2Config =

    /// Validate dimensions and constants required by the implemented architecture.
    let validate (config: DistilGpt2Config) =
        if
            config.VocabSize <= 0L
            || config.EmbeddingSize <= 0L
            || config.IntermediateSize <= 0L
            || config.NumHiddenLayers <= 0
            || config.NumAttentionHeads <= 0L
            || config.MaxPositionEmbeddings <= 0L
            || config.LayerNormEps <= 0.0
            || not (Double.IsFinite config.LayerNormEps)
        then
            invalidArg (nameof config) "DistilGPT-2 dimensions and constants must be finite and positive."

        if config.EmbeddingSize % config.NumAttentionHeads <> 0L then
            invalidArg (nameof config) "Embedding size must be divisible by the number of attention heads."

        if
            config.BosTokenId < 0L
            || config.BosTokenId >= config.VocabSize
        then
            invalidArg (nameof config) "BOS token ID must be within the vocabulary."

        if
            config.EosTokenId < 0L
            || config.EosTokenId >= config.VocabSize
        then
            invalidArg (nameof config) "EOS token ID must be within the vocabulary."

/// A GPT-2 Conv1D projection stored in Hugging Face [input, output] weight layout.
type DistilGpt2Conv1D = {
    Weight: Tensor
    Bias: Tensor
} with

    /// Apply the affine projection without transposing its stored weight.
    member this.forward(input: Tensor) = input.matmul this.Weight + this.Bias

/// Projection layers in one DistilGPT-2 self-attention block.
type DistilGpt2Attention = {
    Qkv: DistilGpt2Conv1D
    Output: DistilGpt2Conv1D
}

/// Projection layers in one DistilGPT-2 feed-forward block.
type DistilGpt2Mlp = {
    Input: DistilGpt2Conv1D
    Output: DistilGpt2Conv1D
}

/// One DistilGPT-2 transformer block.
type DistilGpt2Block = {
    Norm1: LayerNorm
    Attention: DistilGpt2Attention
    Norm2: LayerNorm
    Mlp: DistilGpt2Mlp
}
