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
    let causalLm = DistilGpt2.asCausalLm model
    let options = GenerationOptions.greedy maxNewTokens

    seq {
        use session = Generation.createSession options promptTokenIds causalLm
        let decoder = tokenizer.createDecoder ()

        yield tokenizer.decode promptTokenIds

        while not session.IsFinished do
            match session.Step() with
            | None -> ()
            | Some tokenId when Set.contains tokenId causalLm.EosTokenIds -> ()
            | Some tokenId ->
                let text = decoder.append tokenId

                if text.Length > 0 then
                    yield text

        let remaining = decoder.complete ()

        if remaining.Length > 0 then
            yield remaining
    }

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

        generate model tokenizer maxNewTokens prompt
        |> Seq.iter (fun text ->
            printf "%s" text
            Console.Out.Flush())

        printfn ""
        0
    finally
        DistilGpt2.dispose model
