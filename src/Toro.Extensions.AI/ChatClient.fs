namespace Toro.Extensions.AI

open System
open System.Threading
open System.Threading.Tasks
open Microsoft.Extensions.AI
open Toro.Models
open Toro.Text

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
    /// Optional factory for request-local incremental decoding.
    CreateDecoder: (unit -> IncrementalDecoder) option
    /// Maximum number of generated tokens when a request does not specify a limit.
    DefaultMaxOutputTokens: int
}

module private Request =

    type Prepared = {
        PromptTokenIds: int64 list
        GenerationOptions: GenerationOptions
    }

    let private validateMessage index (message: ChatMessage) =
        if isNull message then
            invalidArg "messages" $"Chat message at index {index} is null."

        let supportedRole =
            [ ChatRole.System; ChatRole.User; ChatRole.Assistant ]
            |> List.contains message.Role

        if not supportedRole then
            invalidOp $"Chat role '{message.Role}' at index {index} is not supported."

        if isNull message.Contents then
            invalidArg "messages" $"Chat message contents at index {index} are null."

        message.Contents
        |> Seq.tryFind (function
            | :? TextContent -> false
            | _ -> true)
        |> Option.iter (function
            | null -> invalidOp $"Null chat content at message index {index} is not supported."
            | content -> invalidOp $"Chat content '{content.GetType().Name}' at message index {index} is not supported.")

    let private sampling (options: ChatOptions) =
        match Option.ofNullable options.Temperature with
        | None
        | Some 0.0f -> Greedy
        | Some temperature when temperature > 0.0f && Single.IsFinite temperature -> Temperature(float temperature)
        | Some _ -> invalidOp "Chat temperature must be finite and non-negative."

    let private validateOptions modelId (options: ChatOptions) =
        if
            not (String.IsNullOrWhiteSpace options.ModelId)
            && not (String.Equals(options.ModelId, modelId, StringComparison.Ordinal))
        then
            invalidOp $"Requested model '{options.ModelId}' does not match chat client model '{modelId}'."

        [
            "TopP", options.TopP.HasValue
            "TopK", options.TopK.HasValue
            "FrequencyPenalty", options.FrequencyPenalty.HasValue
            "PresencePenalty", options.PresencePenalty.HasValue
            "Seed", options.Seed.HasValue
            "ResponseFormat",
            not (isNull options.ResponseFormat)
            && options.ResponseFormat <> ChatResponseFormat.Text
            "StopSequences",
            not (isNull options.StopSequences)
            && options.StopSequences.Count > 0
            "Tools", not (isNull options.Tools) && options.Tools.Count > 0
            "ToolMode", not (isNull options.ToolMode)
            "ConversationId", not (String.IsNullOrWhiteSpace options.ConversationId)
            "AdditionalProperties",
            not (isNull options.AdditionalProperties)
            && options.AdditionalProperties.Count > 0
        ]
        |> List.tryFind snd
        |> Option.iter (fun (name, _) -> invalidOp $"Chat option '{name}' is not supported by Toro.Extensions.AI.")

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
            options.MaxOutputTokens
            |> Option.ofNullable
            |> Option.defaultValue config.DefaultMaxOutputTokens

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

    let withoutEos eosTokenIds tokenIds =
        match List.rev tokenIds with
        | tokenId :: rest when Set.contains tokenId eosTokenIds -> List.rev rest, true
        | _ -> tokenIds, false

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

module private Streaming =

    let private role index =
        if index = 0 then
            Nullable ChatRole.Assistant
        else
            Nullable()

    let responses config (prepared: Request.Prepared) cancellationToken =
        seq {
            let options = {
                prepared.GenerationOptions with
                    CancellationToken = cancellationToken
            }

            use session = Generation.createSession options prepared.PromptTokenIds config.Model

            let decoder =
                match config.CreateDecoder with
                | Some create -> create ()
                | None -> IncrementalDecoder.fromDecode config.Decode

            let updates =
                session.Tokens
                |> Seq.filter (fun tokenId -> not (Set.contains tokenId config.Model.EosTokenIds))
                |> decoder.appendAll
                |> Seq.mapi (fun index text ->
                    cancellationToken.ThrowIfCancellationRequested()
                    Response.update config.ModelId (role index) (Nullable()) text)
                |> Seq.cache

            yield! updates

            let finishReason =
                session.GeneratedTokenIds
                |> List.tryLast
                |> Option.exists (fun tokenId -> Set.contains tokenId config.Model.EosTokenIds)
                |> Response.finishReason

            yield Response.update config.ModelId (role (Seq.length updates)) (Nullable finishReason) ""
        }

type private ToroChatClient<'Cache>(initialConfig: CausalLmChatClientConfig<'Cache>) as this =
    let callbackLock = obj ()

    let config = {
        initialConfig with
            FormatPrompt = fun messages -> lock callbackLock (fun () -> initialConfig.FormatPrompt messages)
            Encode = fun text -> lock callbackLock (fun () -> initialConfig.Encode text)
            Decode = fun tokenIds -> lock callbackLock (fun () -> initialConfig.Decode tokenIds)
            CreateDecoder =
                initialConfig.CreateDecoder
                |> Option.map (fun create () -> lock callbackLock (fun () -> create ()))
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

                    let responseTokenIds, eos = Response.withoutEos config.Model.EosTokenIds tokenIds

                    let text = config.Decode responseTokenIds

                    if isNull text then
                        invalidOp "The token decoder returned null."

                    Response.create config.ModelId eos text),
                cancellationToken
            )

        member _.GetStreamingResponseAsync(messages, options, cancellationToken) =
            let prepared = Request.prepare config messages options cancellationToken

            AsyncEnumerable.ofBackgroundSeq cancellationToken (Streaming.responses config prepared)

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
