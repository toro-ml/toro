open System
open System.IO
open TorchSharp
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
            SpecialTokens = [ "<|endoftext|>", 0; "<|im_start|>", 1; "<|im_end|>", 2 ]
            UnknownToken = Some "<|endoftext|>"
            PreTokenizer = ByteLevelPreTokenizer
    }

let chatPrompt userPrompt =
    $"<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n<|im_start|>user\n{userPrompt}<|im_end|>\n<|im_start|>assistant\n"

let generate (model: SmolLm2) (tokenizer: Tokenizer) maxNewTokens userPrompt =
    let promptTokenIds = tokenizer.encode (chatPrompt userPrompt) |> List.map int64

    model
    |> SmolLm2.asCausalLm
    |> Generation.generate (GenerationOptions.greedy maxNewTokens) promptTokenIds
    |> List.map int
    |> tokenizer.decode

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
        printfn "%s" (generate model tokenizer maxNewTokens prompt)
        0
    finally
        SmolLm2.dispose model
