open System
open System.IO
open TorchSharp
open Toro.Hub
open Toro.Models
open Toro.Text

let repoId = "distilbert/distilgpt2"
let revision = "2290a62682d06624634c1f46a6ad5be0f47f38aa"

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

let loadTokenizer eosTokenId =
    let paths =
        [ "vocab.json"; "merges.txt" ]
        |> List.map (hubFile >> Hub.download)
        |> Async.Parallel
        |> Async.RunSynchronously

    Tokenizer.fromBpe {
        BpeConfig.create paths[0] paths[1] with
            ByteLevel = true
            SpecialTokens = [ "<|endoftext|>", eosTokenId ]
            UnknownToken = Some "<|endoftext|>"
            PreTokenizer = ByteLevelPreTokenizer
    }

let generate (model: DistilGpt2) (tokenizer: Tokenizer) maxNewTokens prompt =
    let promptTokenIds = tokenizer.encode prompt

    let generatedTokenIds =
        model
        |> DistilGpt2.asCausalLm
        |> Generation.generate (GenerationOptions.greedy maxNewTokens) promptTokenIds

    List.append promptTokenIds generatedTokenIds
    |> tokenizer.decode

[<EntryPoint>]
let main argv =
    let prompt =
        Array.tryItem 0 argv
        |> Option.defaultValue "The F# programming language"

    let maxNewTokens =
        Array.tryItem 1 argv
        |> Option.map int
        |> Option.defaultValue 24

    printfn "Loading %s at %s ..." repoId revision
    let directory = downloadModel ()
    let model, report = DistilGpt2.loadFromDirectory directory torch.CPU

    try
        let tokenizer = loadTokenizer model.Config.EosTokenId
        printfn "Loaded %d tensors; ignored %d checkpoint buffers." report.Loaded.Length report.Ignored.Length
        printfn "Prompt: %s" prompt
        printfn ""
        printfn "%s" (generate model tokenizer maxNewTokens prompt)
        0
    finally
        DistilGpt2.dispose model
