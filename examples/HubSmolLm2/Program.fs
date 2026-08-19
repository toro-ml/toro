open System
open System.IO
open System.Threading
open Microsoft.Extensions.AI
open TorchSharp
open Toro.Extensions.AI
open Toro.Hub
open Toro.Models
open Toro.Text

let repoId = "HuggingFaceTB/SmolLM2-135M-Instruct"
let revision = "12fd25f77366fa6b3b4b768ec3050bf629380bac"

let hubFile filename = {
    RepoId = repoId
    Revision = revision
    Filename = filename
}

let downloadModel () =
    let paths =
        [ "config.json"; "model.safetensors" ]
        |> List.map (hubFile >> Hub.download)
        |> Async.Parallel
        |> Async.RunSynchronously

    Path.GetDirectoryName paths[0]

let loadTokenizer () =
    let paths =
        [ "vocab.json"; "merges.txt" ]
        |> List.map (hubFile >> Hub.download)
        |> Async.Parallel
        |> Async.RunSynchronously

    Tokenizer.fromBpe {
        BpeConfig.create paths[0] paths[1] with
            ByteLevel = true
            SpecialTokens = [ "<|endoftext|>", 0L; "<|im_start|>", 1L; "<|im_end|>", 2L ]
            UnknownToken = Some "<|endoftext|>"
            PreTokenizer = ByteLevelPreTokenizer
    }

let formatPrompt (messages: ChatMessage list) =
    let messages =
        if
            messages
            |> List.exists (fun message -> message.Role = ChatRole.System)
        then
            messages
        else
            ChatMessage(ChatRole.System, "You are a helpful AI assistant named SmolLM, trained by Hugging Face")
            :: messages

    messages
    |> List.map (fun message -> $"<|im_start|>{message.Role.Value}\n{message.Text}<|im_end|>\n")
    |> String.concat ""
    |> fun prompt -> prompt + "<|im_start|>assistant\n"

let generate (model: SmolLm2) (tokenizer: Tokenizer) maxNewTokens (userPrompt: string) =
    task {
        use client =
            CausalLmChatClient.create {
                ModelId = repoId
                Model = SmolLm2.asCausalLm model
                FormatPrompt = formatPrompt
                Encode = tokenizer.encode
                Decode = tokenizer.decode
                DefaultMaxOutputTokens = 32
            }

        let options = ChatOptions()
        options.MaxOutputTokens <- Nullable maxNewTokens

        let updates =
            client.GetStreamingResponseAsync([ ChatMessage(ChatRole.User, userPrompt) ], options, CancellationToken.None)

        use enumerator = updates.GetAsyncEnumerator()
        let mutable hasNext = true

        while hasNext do
            let! next = enumerator.MoveNextAsync()
            hasNext <- next

            if hasNext then
                printf "%s" enumerator.Current.Text

        printfn ""
    }

[<EntryPoint>]
let main argv =
    let prompt =
        Array.tryItem 0 argv
        |> Option.defaultValue "What is 84 * 3 / 2?"

    let maxNewTokens =
        Array.tryItem 1 argv
        |> Option.map int
        |> Option.defaultValue 32

    printfn "Loading %s at %s ..." repoId revision
    let tokenizer = loadTokenizer ()
    let directory = downloadModel ()
    let model, report = SmolLm2.loadFromDirectory directory torch.CPU

    try
        printfn "Loaded %d tensors." report.Loaded.Length
        printfn "Prompt: %s" prompt
        printfn ""

        generate model tokenizer maxNewTokens prompt
        |> _.GetAwaiter().GetResult()

        0
    finally
        SmolLm2.dispose model
