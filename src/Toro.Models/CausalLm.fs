namespace Toro.Models

open System
open System.Threading
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

/// Token-selection strategy used during generation.
type GenerationSampling =
    /// Select the highest-logit token.
    | Greedy
    /// Sample from logits divided by a positive temperature.
    | Temperature of temperature: float

/// Options for one causal language-model generation session.
type GenerationOptions = {
    /// Maximum number of tokens to generate after the prompt.
    MaxNewTokens: int
    /// Token-selection strategy applied to each next-token distribution.
    Sampling: GenerationSampling
    /// Cancellation signal checked before each model invocation.
    CancellationToken: CancellationToken
}

/// Constructors and validation for generation options.
module GenerationOptions =

    /// Create greedy generation options without cancellation.
    let greedy maxNewTokens = {
        MaxNewTokens = maxNewTokens
        Sampling = Greedy
        CancellationToken = CancellationToken.None
    }

    /// Create temperature-sampling options without cancellation.
    let temperature temperature maxNewTokens = {
        MaxNewTokens = maxNewTokens
        Sampling = Temperature temperature
        CancellationToken = CancellationToken.None
    }

    let internal validate options =
        if options.MaxNewTokens < 0 then
            invalidArg (nameof options) "Maximum new-token count must be non-negative."

        match options.Sampling with
        | Greedy -> ()
        | Temperature value when value > 0.0 && Double.IsFinite value -> ()
        | Temperature _ -> invalidArg (nameof options) "Sampling temperature must be finite and positive."

/// A single-request generation session that owns its token sequence and key/value cache.
type GenerationSession<'Cache> internal (model: CausalLm<'Cache>, promptTokenIds: int64 list, options: GenerationOptions) =
    let prompt = promptTokenIds |> List.toArray

    let capacity =
        GenerationOptions.validate options

        if prompt.Length = 0 then
            invalidArg (nameof promptTokenIds) "Generation prompt must contain at least one token."

        let capacity = int64 prompt.Length + int64 options.MaxNewTokens

        if capacity > model.ContextLength then
            invalidArg (nameof options) $"Prompt and generated tokens exceed the model context length of {model.ContextLength}."

        capacity

    let cache = model.CreateCache 1L capacity
    let generated = ResizeArray<int64>()
    let tokens = ResizeArray<int64>(prompt)
    let mutable nextInput = prompt
    let mutable finished = options.MaxNewTokens = 0
    let mutable disposed = false

    let ensureAvailable () =
        if disposed then
            raise (ObjectDisposedException(nameof GenerationSession))

    let selectToken (logits: Tensor) =
        match options.Sampling with
        | Greedy -> logits.argmax(0L).ToInt64()
        | Temperature temperature ->
            let probabilities =
                (logits.to_type (torch.float32) / scalar temperature).softmax (0L)

            torch.multinomial(probabilities, 1L).ToInt64()

    /// Tokens supplied as the prompt.
    member _.PromptTokenIds = prompt |> Array.toList

    /// Tokens generated so far, including a terminating EOS token when one was selected.
    member _.GeneratedTokenIds = generated |> Seq.toList

    /// Prompt and generated tokens accumulated by this session.
    member _.TokenIds = tokens |> Seq.toList

    /// Whether EOS or the configured maximum token count has been reached.
    member _.IsFinished = finished

    /// Generate at most one token. Returns None after the session has finished.
    member _.Step() : int64 option =
        ensureAvailable ()

        if finished then
            None
        else
            options.CancellationToken.ThrowIfCancellationRequested()

            let nextToken =
                Toro.inferenceMode (fun () ->
                    scoped {
                        let inputIds =
                            (torch.tensor (nextInput, dtype = torch.int64, device = model.Device)).unsqueeze (0L)

                        let input = {
                            InputIds = inputIds
                            AttentionMask = None
                            PositionIds = None
                            Cache = Some cache
                        }

                        let output =
                            if model.CacheLength cache = 0L then
                                CausalLm.prefill input model
                            else
                                CausalLm.decode input model

                        return output.Logits.at [ I 0; I -1 ] |> selectToken
                    })

            generated.Add nextToken
            tokens.Add nextToken
            nextInput <- [| nextToken |]

            if
                Set.contains nextToken model.EosTokenIds
                || generated.Count = options.MaxNewTokens
            then
                finished <- true

            Some nextToken

    /// Generate until EOS or the configured maximum token count is reached.
    member this.Generate() : int64 list =
        ensureAvailable ()

        while not this.IsFinished do
            this.Step() |> ignore

        this.GeneratedTokenIds

    interface IDisposable with
        member _.Dispose() =
            if not disposed then
                disposed <- true
                model.DisposeCache cache

/// Session-based causal language-model generation.
module Generation =

    /// Create a request-local generation session. The caller owns the returned session.
    let createSession options promptTokenIds model =
        new GenerationSession<_>(model, promptTokenIds, options)

    /// Generate token IDs and dispose the request-local cache before returning.
    let generate options promptTokenIds model =
        use session = createSession options promptTokenIds model
        session.Generate()
