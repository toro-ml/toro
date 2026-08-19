namespace Toro.Models

open TorchSharp
open Toro

/// Tensor input shared by cached causal language models.
type CausalLmInput<'Cache> = {
    /// Token IDs with shape [batch, sequence].
    InputIds: Tensor
    /// Optional 0/1 padding mask with shape [batch, total sequence].
    AttentionMask: Tensor option
    /// Optional absolute positions with shape [sequence] or [batch, sequence].
    PositionIds: Tensor option
    /// Optional model-specific key/value cache.
    Cache: 'Cache option
}

/// Tensor output shared by cached causal language models.
type CausalLmOutput<'Cache> = {
    /// Vocabulary logits with shape [batch, sequence, vocabulary].
    Logits: Tensor
    /// The cache supplied in the input, after the successful forward pass.
    Cache: 'Cache option
}

/// Typed operations and metadata required for causal language-model generation.
type CausalLm<'Cache> = {
    /// Maximum total number of prompt and generated tokens.
    ContextLength: int64
    /// Token IDs that terminate generation after being selected.
    EosTokenIds: Set<int64>
    /// Device on which token inputs are created.
    Device: torch.Device
    /// Allocate an empty request-local cache for a batch and token capacity.
    CreateCache: int64 -> int64 -> 'Cache
    /// Return the number of tokens currently stored by a cache.
    CacheLength: 'Cache -> int64
    /// Release a request-local cache and its tensors.
    DisposeCache: 'Cache -> unit
    /// Run the model for a prefill or single-token decode input.
    Forward: CausalLmInput<'Cache> -> CausalLmOutput<'Cache>
}

/// Tensor-level prefill and decode operations for causal language models.
module CausalLm =

    /// Populate an empty cache from one or more input tokens.
    let prefill (input: CausalLmInput<'Cache>) (model: CausalLm<'Cache>) =
        match input.Cache with
        | None -> invalidArg (nameof input) "Causal LM prefill requires a cache."
        | Some cache when model.CacheLength cache <> 0L ->
            invalidArg (nameof input) "Causal LM prefill requires an empty cache."
        | Some _ -> model.Forward input

    /// Decode exactly one new token using a populated cache.
    let decode (input: CausalLmInput<'Cache>) (model: CausalLm<'Cache>) =
        match input.Cache with
        | None -> invalidArg (nameof input) "Causal LM decode requires a cache."
        | Some cache when model.CacheLength cache = 0L ->
            invalidArg (nameof input) "Causal LM decode requires a populated cache."
        | Some _ when
            input.InputIds.shape.Length <> 2
            || input.InputIds.shape[1] <> 1L
            ->
            invalidArg (nameof input) "Causal LM decode requires input IDs with shape [batch, 1]."
        | Some _ -> model.Forward input
