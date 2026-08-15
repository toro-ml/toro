module ChatClientTests

open System
open System.Collections.Generic
open System.Threading
open Microsoft.Extensions.AI
open TorchSharp
open Toro.Extensions.AI
open Toro.Models
open Xunit

type private FakeCache = {
    mutable Length: int64
    mutable Generated: int
    mutable Disposed: bool
}

let private createClient () =
    let mutable lastCache: FakeCache option = None
    let mutable formattedPrompt = ""

    let model = {
        ContextLength = 16L
        EosTokenIds = Set.singleton 2L
        Device = torch.CPU
        CreateCache =
            fun _ _ ->
                let cache = {
                    Length = 0L
                    Generated = 0
                    Disposed = false
                }

                lastCache <- Some cache
                cache
        CacheLength = _.Length
        DisposeCache = fun cache -> cache.Disposed <- true
        Forward =
            fun input ->
                let cache = input.Cache |> Option.get
                let tokenId = if cache.Generated = 0 then 1 else 2
                cache.Generated <- cache.Generated + 1
                cache.Length <- cache.Length + input.InputIds.shape[1]

                let values = Array.create 4 Single.NegativeInfinity
                values[tokenId] <- 0.0f
                let tokenLogits = torch.tensor values

                let logits =
                    tokenLogits.reshape([| 1L; 1L; 4L |]).expand ([| 1L; input.InputIds.shape[1]; 4L |])

                { Logits = logits; Cache = Some cache }
    }

    let config = {
        ModelId = "fake-chat"
        Model = model
        FormatPrompt =
            fun messages ->
                let prompt =
                    messages
                    |> List.map (fun message -> $"{message.Role.Value}:{message.Text}")
                    |> String.concat "|"

                formattedPrompt <- prompt
                prompt
        Encode = fun _ -> [ 3L ]
        Decode =
            function
            | [] -> ""
            | [ 1L ] -> "hello"
            | tokenIds -> invalidOp $"Unexpected generated tokens: {tokenIds}."
        DefaultMaxOutputTokens = 4
    }

    CausalLmChatClient.create config, (fun () -> formattedPrompt), (fun () -> lastCache |> Option.get)

[<Fact>]
let ``response applies text chat options and reports model metadata`` () =
    let client, prompt, _ = createClient ()
    use client = client
    let options = ChatOptions()
    options.Instructions <- "be concise"
    options.MaxOutputTokens <- Nullable 4
    options.Temperature <- Nullable 0.8f

    let response =
        client
            .GetResponseAsync([ ChatMessage(ChatRole.User, "question") ], options, CancellationToken.None)
            .GetAwaiter()
            .GetResult()

    Assert.Equal("hello", response.Text)
    Assert.Equal("fake-chat", response.ModelId)
    Assert.Equal(Nullable ChatFinishReason.Stop, response.FinishReason)
    Assert.Equal("system:be concise|user:question", prompt ())

    let metadata =
        client.GetService(typeof<ChatClientMetadata>, null) :?> ChatClientMetadata

    Assert.Equal("Toro", metadata.ProviderName)
    Assert.Equal("fake-chat", metadata.DefaultModelId)

[<Fact>]
let ``streaming emits text and releases its request cache`` () =
    let client, _, cache = createClient ()
    use client = client

    let updates =
        client.GetStreamingResponseAsync([ ChatMessage(ChatRole.User, "question") ], null, CancellationToken.None)

    let enumerator = updates.GetAsyncEnumerator()
    let received = ResizeArray<ChatResponseUpdate>()

    try
        let mutable hasNext = enumerator.MoveNextAsync().AsTask().GetAwaiter().GetResult()

        while hasNext do
            received.Add enumerator.Current
            hasNext <- enumerator.MoveNextAsync().AsTask().GetAwaiter().GetResult()
    finally
        enumerator.DisposeAsync().AsTask().GetAwaiter().GetResult()

    Assert.Equal("hello", received |> Seq.map _.Text |> String.concat "")
    Assert.Equal(Nullable ChatRole.Assistant, received[0].Role)
    Assert.Equal(Nullable ChatFinishReason.Stop, received[received.Count - 1].FinishReason)
    Assert.True((cache ()).Disposed)

[<Fact>]
let ``tool roles and non-text content are rejected`` () =
    let client, _, _ = createClient ()
    use client = client

    Assert.Throws<InvalidOperationException>(fun () ->
        client.GetResponseAsync([ ChatMessage(ChatRole.Tool, "result") ], null, CancellationToken.None)
        |> ignore)
    |> ignore

    let contents = ResizeArray<AIContent>()
    contents.Add(DataContent(ReadOnlyMemory<byte>([| 1uy |]), "image/png"))

    Assert.Throws<InvalidOperationException>(fun () ->
        client.GetResponseAsync([ ChatMessage(ChatRole.User, contents) ], null, CancellationToken.None)
        |> ignore)
    |> ignore
