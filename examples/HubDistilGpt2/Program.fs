open System
open TorchSharp
open Toro
open Toro.Hub
open Toro.NN
open Toro.Text

let repoId = "distilbert/distilgpt2"
let revision = "2290a62682d06624634c1f46a6ad5be0f47f38aa"
let embeddingSize = 768L
let intermediateSize = 3072L
let numHeads = 12L
let numLayers = 6
let vocabSize = 50257L
let maxPositions = 1024L
let eosTokenId = 50256L

type Gpt2Attention = {
    Qkv: Linear
    Output: Linear
    NumHeads: int64
} with

    member this.forward(input: Tensor) : Tensor =
        let batchSize = input.shape[0]
        let sequenceLength = input.shape[1]
        let headSize = embeddingSize / this.NumHeads
        let qkv = this.Qkv.forward input
        let chunks = qkv.chunk (3L, -1L)

        let toHeads (tensor: Tensor) =
            tensor.reshape([| batchSize; sequenceLength; this.NumHeads; headSize |]).permute ([| 0L; 2L; 1L; 3L |])

        let query = toHeads chunks[0]
        let key = toHeads chunks[1]
        let value = toHeads chunks[2]

        let attended =
            torch.nn.functional.scaled_dot_product_attention (query, key, value, is_casual = true)

        attended.permute([| 0L; 2L; 1L; 3L |]).contiguous().reshape ([| batchSize; sequenceLength; embeddingSize |])
        |> this.Output.forward

type Gpt2Mlp = {
    Input: Linear
    Output: Linear
} with

    member this.forward(input: Tensor) : Tensor =
        let hidden = this.Input.forward input

        let activated =
            let cubic = hidden.pow (scalar 3.0)
            let inner = hidden + cubic * scalar 0.044715
            let cdf = (inner * scalar (sqrt (2.0 / Math.PI))).tanh () + scalar 1.0
            hidden * scalar 0.5 * cdf

        this.Output.forward activated

type Gpt2Block = {
    Norm1: LayerNorm
    Attention: Gpt2Attention
    Norm2: LayerNorm
    Mlp: Gpt2Mlp
} with

    member this.forward(input: Tensor) : Tensor =
        let attended = input |> this.Norm1.forward |> this.Attention.forward
        let hidden = input + attended
        let projected = hidden |> this.Norm2.forward |> this.Mlp.forward
        hidden + projected

type DistilGpt2 = {
    TokenEmbedding: Embedding
    PositionEmbedding: Embedding
    Blocks: Gpt2Block list
    FinalNorm: LayerNorm
} with

    member this.forward(tokens: Tensor) : Tensor =
        let sequenceLength = tokens.shape[1]

        if sequenceLength > maxPositions then
            invalidArg (nameof tokens) $"DistilGPT2 accepts at most {maxPositions} tokens."

        let positions =
            torch.arange (sequenceLength, dtype = torch.int64, device = torch.CPU)
            |> _.unsqueeze(0L)

        let hidden =
            this.TokenEmbedding.forward tokens
            + this.PositionEmbedding.forward positions

        let hidden =
            this.Blocks
            |> List.fold (fun state block -> block.forward state) hidden

        let hidden = this.FinalNorm.forward hidden
        hidden.matmul (this.TokenEmbedding.Embeddings.t ())

let createLayerNorm () =
    LayerNorm.init
        embeddingSize
        {
            LayerNormConfig.defaultConfig with
                Eps = 1e-5
        }
        torch.float32
        torch.CPU

let createBlock () = {
    Norm1 = createLayerNorm ()
    Attention = {
        Qkv = Linear.init embeddingSize (3L * embeddingSize) torch.float32 torch.CPU
        Output = Linear.init embeddingSize embeddingSize torch.float32 torch.CPU
        NumHeads = numHeads
    }
    Norm2 = createLayerNorm ()
    Mlp = {
        Input = Linear.init embeddingSize intermediateSize torch.float32 torch.CPU
        Output = Linear.init intermediateSize embeddingSize torch.float32 torch.CPU
    }
}

let createModel () = {
    TokenEmbedding = Embedding.init vocabSize embeddingSize torch.float32 torch.CPU
    PositionEmbedding = Embedding.init maxPositions embeddingSize torch.float32 torch.CPU
    Blocks = List.init numLayers (fun _ -> createBlock ())
    FinalNorm = createLayerNorm ()
}

let nameMapping =
    let block sourceSuffix targetSuffix =
        NameRule.rewrite ("transformer.h.{block}." + sourceSuffix) ("Blocks.{block}." + targetSuffix)

    NameMapping.create [
        NameRule.ignoreSuffix "attn.bias"
        NameRule.rename "transformer.wte.weight" "TokenEmbedding.Embeddings"
        NameRule.rename "transformer.wpe.weight" "PositionEmbedding.Embeddings"
        block "ln_1.weight" "Norm1.Weight"
        block "ln_1.bias" "Norm1.Bias"
        block "attn.c_attn.weight" "Attention.Qkv.Weight"
        block "attn.c_attn.bias" "Attention.Qkv.Bias"
        block "attn.c_proj.weight" "Attention.Output.Weight"
        block "attn.c_proj.bias" "Attention.Output.Bias"
        block "ln_2.weight" "Norm2.Weight"
        block "ln_2.bias" "Norm2.Bias"
        block "mlp.c_fc.weight" "Mlp.Input.Weight"
        block "mlp.c_fc.bias" "Mlp.Input.Bias"
        block "mlp.c_proj.weight" "Mlp.Output.Weight"
        block "mlp.c_proj.bias" "Mlp.Output.Bias"
        NameRule.rename "transformer.ln_f.weight" "FinalNorm.Weight"
        NameRule.rename "transformer.ln_f.bias" "FinalNorm.Bias"
    ]

let hubFile filename = {
    RepoId = repoId
    Revision = revision
    Filename = filename
}

let isConv1dWeight (name: string) =
    name.EndsWith(".attn.c_attn.weight", StringComparison.Ordinal)
    || name.EndsWith(".attn.c_proj.weight", StringComparison.Ordinal)
    || name.EndsWith(".mlp.c_fc.weight", StringComparison.Ordinal)
    || name.EndsWith(".mlp.c_proj.weight", StringComparison.Ordinal)

let loadModel () =
    let model = createModel ()

    let weightPath =
        Hub.download (hubFile "model.safetensors")
        |> Async.RunSynchronously

    let report =
        scoped {
            let weights = SafeTensors.load weightPath

            let transformed =
                weights
                |> Map.map (fun name tensor -> if isConv1dWeight name then tensor.t () else tensor)

            return
                transformed
                |> Model.loadFromDictWith nameMapping Strict model
        }

    model, report

let loadTokenizer () =
    let paths =
        [ "vocab.json"; "merges.txt" ]
        |> List.map (hubFile >> Hub.download)
        |> Async.Parallel
        |> Async.RunSynchronously

    Tokenizer.fromBpe {
        BpeConfig.create paths[0] paths[1] with
            ByteLevel = true
            SpecialTokens = [ "<|endoftext|>", int eosTokenId ]
            UnknownToken = Some "<|endoftext|>"
            PreTokenizer = ByteLevelPreTokenizer
    }

let generate (model: DistilGpt2) (tokenizer: Tokenizer) maxNewTokens prompt =
    let generated = ResizeArray<int64>(tokenizer.encode prompt |> List.map int64)

    if generated.Count = 0 then
        invalidArg (nameof prompt) "The prompt must produce at least one token."

    if int64 generated.Count + int64 maxNewTokens > maxPositions then
        invalidArg (nameof maxNewTokens) $"The prompt and generated text must fit within {maxPositions} tokens."

    let mutable finished = false

    for _ in 1..maxNewTokens do
        if not finished then
            let nextToken =
                Toro.inferenceMode (fun () ->
                    scoped {
                        let input =
                            torch.tensor (generated.ToArray(), dtype = torch.int64, device = torch.CPU)

                        let logits = model.forward (input.unsqueeze 0L)
                        return (logits.at [ I 0; I -1 ]).argmax(0L).ToInt64()
                    })

            generated.Add nextToken
            finished <- nextToken = eosTokenId

    generated |> Seq.map int |> Seq.toList |> tokenizer.decode

[<EntryPoint>]
let main argv =
    let prompt =
        Array.tryItem 0 argv
        |> Option.defaultValue "The F# programming language"

    let maxNewTokens =
        Array.tryItem 1 argv
        |> Option.map int
        |> Option.defaultValue 24

    if maxNewTokens < 0 then
        invalidArg (nameof argv) "max-new-tokens must be non-negative."

    printfn "Loading %s at %s ..." repoId revision
    let tokenizer = loadTokenizer ()
    let model, report = loadModel ()
    printfn "Loaded %d tensors; ignored %d checkpoint buffers." report.Loaded.Length report.Ignored.Length
    printfn "Prompt: %s" prompt
    printfn ""
    printfn "%s" (generate model tokenizer maxNewTokens prompt)
    0
