open System
open System.IO
open TorchSharp
open Toro
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
    let promptTokens = tokenizer.encode (chatPrompt userPrompt) |> List.map int64

    if promptTokens.IsEmpty then
        invalidArg (nameof userPrompt) "The prompt must produce at least one token."

    if int64 promptTokens.Length + int64 maxNewTokens > model.Config.MaxPositionEmbeddings then
        invalidArg
            (nameof maxNewTokens)
            $"The prompt and generated text must fit within {model.Config.MaxPositionEmbeddings} tokens."

    use cache =
        SmolLm2.createCache 1L (int64 promptTokens.Length + int64 maxNewTokens) model

    let response = ResizeArray<int>()
    let mutable inputIds = promptTokens |> List.toArray
    let mutable finished = false

    for _ in 1..maxNewTokens do
        if not finished then
            let nextToken =
                Toro.inferenceMode (fun () ->
                    scoped {
                        let input = torch.tensor (inputIds, dtype = torch.int64, device = torch.CPU)

                        let output =
                            model
                            |> SmolLm2.forward {
                                InputIds = input.unsqueeze 0L
                                AttentionMask = None
                                PositionIds = None
                                Cache = Some cache
                            }

                        return (output.Logits.at [ I 0; I -1 ]).argmax(0L).ToInt64()
                    })

            if nextToken = model.Config.EosTokenId then
                finished <- true
            else
                inputIds <- [| nextToken |]
                response.Add(int nextToken)

    response |> Seq.toList |> tokenizer.decode

[<EntryPoint>]
let main argv =
    let prompt =
        Array.tryItem 0 argv
        |> Option.defaultValue "What is 84 * 3 / 2?"

    let maxNewTokens =
        Array.tryItem 1 argv
        |> Option.map int
        |> Option.defaultValue 32

    if maxNewTokens < 0 then
        invalidArg (nameof argv) "max-new-tokens must be non-negative."

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
