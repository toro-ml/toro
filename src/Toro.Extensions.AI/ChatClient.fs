namespace Toro.Extensions.AI

open System
open System.Collections.Generic
open System.Threading
open System.Threading.Tasks
open Microsoft.Extensions.AI
open Toro.Models

/// Configuration for adapting a Toro causal language model to IChatClient.
type CausalLmChatClientConfig<'Cache> = {
    /// Model identifier reported through Microsoft.Extensions.AI responses and metadata.
    ModelId: string
    /// Bound causal language model used for generation.
    Model: CausalLm<'Cache>
    /// Model-specific formatter for validated text-only chat messages.
    FormatPrompt: ChatMessage list -> string
    /// Encode a formatted prompt into token IDs.
    Encode: string -> int64 list
    /// Decode generated token IDs into response text.
    Decode: int64 list -> string
    /// Maximum number of generated tokens when a request does not specify a limit.
    DefaultMaxOutputTokens: int
}

module private Request =

    type Prepared = {
        PromptTokenIds: int64 list
        GenerationOptions: GenerationOptions
    }

    let private unsupported name condition =
        if condition then
            invalidOp $"Chat option '{name}' is not supported by Toro.Extensions.AI."

    let private validateMessage index (message: ChatMessage) =
        if isNull message then
            invalidArg "messages" $"Chat message at index {index} is null."

        if
            message.Role <> ChatRole.System
            && message.Role <> ChatRole.User
            && message.Role <> ChatRole.Assistant
        then
            invalidOp $"Chat role '{message.Role}' at index {index} is not supported."

        if isNull message.Contents then
            invalidArg "messages" $"Chat message contents at index {index} are null."

        for content in message.Contents do
            match content with
            | :? TextContent -> ()
            | null -> invalidOp $"Null chat content at message index {index} is not supported."
            | _ -> invalidOp $"Chat content '{content.GetType().Name}' at message index {index} is not supported."

    let private sampling (options: ChatOptions) =
        if
            not options.Temperature.HasValue
            || options.Temperature.Value = 0.0f
        then
            Greedy
        elif
            options.Temperature.Value > 0.0f
            && Single.IsFinite options.Temperature.Value
        then
            Temperature(float options.Temperature.Value)
        else
            invalidOp "Chat temperature must be finite and non-negative."

    let private validateOptions modelId (options: ChatOptions) =
        if
            not (String.IsNullOrWhiteSpace options.ModelId)
            && not (String.Equals(options.ModelId, modelId, StringComparison.Ordinal))
        then
            invalidOp $"Requested model '{options.ModelId}' does not match chat client model '{modelId}'."

        unsupported "TopP" options.TopP.HasValue
        unsupported "TopK" options.TopK.HasValue
        unsupported "FrequencyPenalty" options.FrequencyPenalty.HasValue
        unsupported "PresencePenalty" options.PresencePenalty.HasValue
        unsupported "Seed" options.Seed.HasValue

        unsupported
            "ResponseFormat"
            (not (isNull options.ResponseFormat)
             && options.ResponseFormat <> ChatResponseFormat.Text)

        unsupported
            "StopSequences"
            (not (isNull options.StopSequences)
             && options.StopSequences.Count > 0)

        unsupported "Tools" (not (isNull options.Tools) && options.Tools.Count > 0)
        unsupported "ToolMode" (not (isNull options.ToolMode))
        unsupported "ConversationId" (not (String.IsNullOrWhiteSpace options.ConversationId))

        unsupported
            "AdditionalProperties"
            (not (isNull options.AdditionalProperties)
             && options.AdditionalProperties.Count > 0)

    let prepare (config: CausalLmChatClientConfig<'Cache>) messages options cancellationToken =
        if isNull messages then
            nullArg (nameof messages)

        let messages = messages |> Seq.toList

        if messages.IsEmpty then
            invalidArg (nameof messages) "At least one chat message is required."

        messages |> List.iteri validateMessage

        let options = if isNull options then ChatOptions() else options

        validateOptions config.ModelId options

        let messages =
            if String.IsNullOrWhiteSpace options.Instructions then
                messages
            else
                ChatMessage(ChatRole.System, options.Instructions)
                :: messages

        let prompt = config.FormatPrompt messages

        if isNull prompt then
            invalidOp "The chat prompt formatter returned null."

        let promptTokenIds = config.Encode prompt

        if isNull (box promptTokenIds) then
            invalidOp "The token encoder returned null."

        if promptTokenIds.IsEmpty then
            invalidOp "The formatted chat prompt produced no token IDs."

        let maxOutputTokens =
            if options.MaxOutputTokens.HasValue then
                options.MaxOutputTokens.Value
            else
                config.DefaultMaxOutputTokens

        if maxOutputTokens < 0 then
            invalidOp "Maximum output-token count must be non-negative."

        if int64 promptTokenIds.Length + int64 maxOutputTokens > config.Model.ContextLength then
            invalidOp $"Prompt and output limit exceed the model context length of {config.Model.ContextLength}."

        {
            PromptTokenIds = promptTokenIds
            GenerationOptions = {
                MaxNewTokens = maxOutputTokens
                Sampling = sampling options
                CancellationToken = cancellationToken
            }
        }

module private Response =

    let finishReason eos =
        if eos then
            ChatFinishReason.Stop
        else
            ChatFinishReason.Length

    let create (modelId: string) eos (text: string) : ChatResponse =
        let message = ChatMessage(ChatRole.Assistant, text)
        let response = ChatResponse(message)
        response.ModelId <- modelId
        response.FinishReason <- Nullable(finishReason eos)
        response

    let update
        (modelId: string)
        (role: Nullable<ChatRole>)
        (finishReason: Nullable<ChatFinishReason>)
        (text: string)
        : ChatResponseUpdate =
        let update = ChatResponseUpdate(role, text)
        update.ModelId <- modelId
        update.FinishReason <- finishReason
        update

type private StreamingEnumerator<'Cache>
    (
        config: CausalLmChatClientConfig<'Cache>,
        prepared: Request.Prepared,
        requestCancellationToken: CancellationToken,
        enumerationCancellationToken: CancellationToken
    ) =

    let cancellation =
        CancellationTokenSource.CreateLinkedTokenSource(requestCancellationToken, enumerationCancellationToken)

    let cancellationToken = cancellation.Token

    let options = {
        prepared.GenerationOptions with
            CancellationToken = cancellationToken
    }

    let session = Generation.createSession options prepared.PromptTokenIds config.Model
    let generated = ResizeArray<int64>()
    let mutable emittedText = ""
    let mutable current = Unchecked.defaultof<ChatResponseUpdate>
    let mutable firstUpdate = true
    let mutable completed = false
    let mutable disposed = false

    let dispose () =
        if not disposed then
            disposed <- true
            (session :> IDisposable).Dispose()
            cancellation.Dispose()

    let nextUpdate () =
        if disposed || completed then
            false
        else
            let mutable ready = false

            while not ready do
                cancellationToken.ThrowIfCancellationRequested()

                if session.IsFinished then
                    current <-
                        Response.update
                            config.ModelId
                            (if firstUpdate then
                                 Nullable ChatRole.Assistant
                             else
                                 Nullable())
                            (Nullable ChatFinishReason.Length)
                            ""

                    firstUpdate <- false
                    completed <- true
                    ready <- true
                else
                    let tokenId = session.Step() |> Option.get
                    let eos = Set.contains tokenId config.Model.EosTokenIds

                    if not eos then
                        generated.Add tokenId

                    let decoded = config.Decode(generated |> Seq.toList)

                    if isNull decoded then
                        invalidOp "The token decoder returned null."

                    if not (decoded.StartsWith(emittedText, StringComparison.Ordinal)) then
                        invalidOp "The token decoder changed text that was already emitted."

                    let stableLength =
                        if session.IsFinished then
                            decoded.Length
                        else
                            let mutable length = decoded.Length

                            while (length > emittedText.Length
                                   && decoded[length - 1] = '\uFFFD') do
                                length <- length - 1

                            length

                    let delta = decoded.Substring(emittedText.Length, stableLength - emittedText.Length)
                    emittedText <- decoded.Substring(0, stableLength)

                    if delta.Length > 0 || session.IsFinished then
                        let finishReason =
                            if session.IsFinished then
                                Nullable(Response.finishReason eos)
                            else
                                Nullable()

                        current <-
                            Response.update
                                config.ModelId
                                (if firstUpdate then
                                     Nullable ChatRole.Assistant
                                 else
                                     Nullable())
                                finishReason
                                delta

                        firstUpdate <- false

                        if session.IsFinished then
                            completed <- true

                        ready <- true

            if completed then
                dispose ()

            true

    let runNextUpdate () =
        try
            nextUpdate ()
        with _ ->
            dispose ()
            reraise ()

    interface IAsyncEnumerator<ChatResponseUpdate> with
        member _.Current = current

        member _.MoveNextAsync() =
            ValueTask<bool>(Task.Run(Func<bool>(runNextUpdate), cancellationToken))

        member _.DisposeAsync() =
            dispose ()
            ValueTask()

type private StreamingEnumerable<'Cache>
    (config: CausalLmChatClientConfig<'Cache>, prepared: Request.Prepared, requestCancellationToken: CancellationToken) =

    interface IAsyncEnumerable<ChatResponseUpdate> with
        member _.GetAsyncEnumerator(enumerationCancellationToken) =
            new StreamingEnumerator<_>(config, prepared, requestCancellationToken, enumerationCancellationToken)
            :> IAsyncEnumerator<ChatResponseUpdate>

type private ToroChatClient<'Cache>(initialConfig: CausalLmChatClientConfig<'Cache>) as this =
    let callbackLock = obj ()

    let config = {
        initialConfig with
            FormatPrompt = fun messages -> lock callbackLock (fun () -> initialConfig.FormatPrompt messages)
            Encode = fun text -> lock callbackLock (fun () -> initialConfig.Encode text)
            Decode = fun tokenIds -> lock callbackLock (fun () -> initialConfig.Decode tokenIds)
    }

    let metadata = ChatClientMetadata("Toro", null, config.ModelId)

    do
        if String.IsNullOrWhiteSpace initialConfig.ModelId then
            invalidArg (nameof initialConfig) "Chat client model ID must not be empty."

        if isNull (box initialConfig.FormatPrompt) then
            invalidArg (nameof initialConfig) "Chat prompt formatter must not be null."

        if isNull (box initialConfig.Encode) then
            invalidArg (nameof initialConfig) "Token encoder must not be null."

        if isNull (box initialConfig.Decode) then
            invalidArg (nameof initialConfig) "Token decoder must not be null."

        if initialConfig.DefaultMaxOutputTokens < 0 then
            invalidArg (nameof initialConfig) "Default maximum output-token count must be non-negative."

    interface IChatClient with
        member _.GetResponseAsync(messages, options, cancellationToken) =
            let prepared = Request.prepare config messages options cancellationToken

            Task.Run(
                Func<ChatResponse>(fun () ->
                    let tokenIds =
                        Generation.generate prepared.GenerationOptions prepared.PromptTokenIds config.Model

                    let eos =
                        tokenIds
                        |> List.tryLast
                        |> Option.exists (fun token -> Set.contains token config.Model.EosTokenIds)

                    let responseTokenIds =
                        if eos then
                            tokenIds |> List.take (tokenIds.Length - 1)
                        else
                            tokenIds

                    let text = config.Decode responseTokenIds

                    if isNull text then
                        invalidOp "The token decoder returned null."

                    Response.create config.ModelId eos text),
                cancellationToken
            )

        member _.GetStreamingResponseAsync(messages, options, cancellationToken) =
            let prepared = Request.prepare config messages options cancellationToken
            StreamingEnumerable(config, prepared, cancellationToken) :> IAsyncEnumerable<ChatResponseUpdate>

        member _.GetService(serviceType, serviceKey) =
            if isNull serviceType then
                nullArg (nameof serviceType)

            if not (isNull serviceKey) then null
            elif serviceType = typeof<ChatClientMetadata> then metadata
            elif serviceType.IsInstanceOfType this then this
            else null

        member _.Dispose() = ()

/// Factory for Microsoft.Extensions.AI chat clients backed by Toro causal language models.
module CausalLmChatClient =

    /// Create a stateless IChatClient. The returned client does not own or dispose the model.
    let create config : IChatClient =
        new ToroChatClient<_>(config) :> IChatClient
