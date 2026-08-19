namespace Toro.Models

open System
open TorchSharp
open Toro

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

    let isFinished () =
        generated.Count >= options.MaxNewTokens
        || (generated.Count > 0
            && Set.contains generated[generated.Count - 1] model.EosTokenIds)

    let nextInput () =
        if generated.Count = 0 then
            prompt
        else
            [| generated[generated.Count - 1] |]

    /// Tokens supplied as the prompt.
    member _.PromptTokenIds = prompt |> Array.toList

    /// Tokens generated so far, including a terminating EOS token when one was selected.
    member _.GeneratedTokenIds = generated |> Seq.toList

    /// Prompt and generated tokens accumulated by this session.
    member _.TokenIds =
        seq {
            yield! prompt
            yield! generated
        }
        |> Seq.toList

    /// Whether EOS or the configured maximum token count has been reached.
    member _.IsFinished = isFinished ()

    /// Generate at most one token. Returns None after the session has finished.
    member _.Step() : int64 option =
        ensureAvailable ()

        if isFinished () then
            None
        else
            options.CancellationToken.ThrowIfCancellationRequested()

            let nextToken =
                Toro.inferenceMode (fun () ->
                    scoped {
                        let inputIds =
                            (torch.tensor (nextInput (), dtype = torch.int64, device = model.Device)).unsqueeze (0L)

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
            Some nextToken

    /// Generate tokens until EOS or the configured maximum token count is reached.
    member this.Tokens =
        Seq.unfold (fun () -> this.Step() |> Option.map (fun tokenId -> tokenId, ())) ()

    /// Generate until EOS or the configured maximum token count is reached.
    member this.Generate() : int64 list =
        this.Tokens |> Seq.iter ignore
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

    /// Enumerate generated token IDs until the session finishes.
    let tokens (session: GenerationSession<'Cache>) = session.Tokens

    /// Generate token IDs and dispose the request-local cache before returning.
    let generate options promptTokenIds model =
        use session = createSession options promptTokenIds model
        session.Generate()
