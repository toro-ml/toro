open System
open TorchSharp
open Toro
open Toro.Hub
open Toro.NN
open Toro.Text

let repoId = "HuggingFaceTB/SmolLM2-135M-Instruct"
let revision = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
let hiddenSize = 576L
let intermediateSize = 1536L
let numHeads = 9L
let numKeyValueHeads = 3L
let numLayers = 30
let vocabSize = 49152L
let maxPositions = 8192L
let ropeTheta = 100000.0
let eosTokenId = 2L
let modelDtype = torch.bfloat16

let rotaryEmbedding (sequenceLength: int64) (headSize: int64) (dtype: torch.ScalarType) =
    let positions =
        torch.arange(scalar (float sequenceLength), dtype = torch.float32, device = torch.CPU).unsqueeze (1L)

    let dimensions =
        torch.arange(scalar (float (headSize / 2L)), dtype = torch.float32, device = torch.CPU).mul (scalar 2.0)

    let inverseFrequencies =
        (dimensions / scalar (float headSize)
         * scalar (log ropeTheta))
            .neg()
            .exp ()

    let frequencies = positions.matmul (inverseFrequencies.unsqueeze (0L))
    let embeddings = torch.cat ([| frequencies; frequencies |], dim = -1L)
    embeddings.cos().to_type (dtype), embeddings.sin().to_type (dtype)

let rotateHalf (input: Tensor) =
    let halves = input.chunk (2L, dim = -1L)
    torch.cat ([| halves[1].neg (); halves[0] |], dim = -1L)

let applyRotaryEmbedding (input: Tensor) (cosines: Tensor) (sines: Tensor) =
    let cosines = cosines.unsqueeze(0L).unsqueeze (0L)
    let sines = sines.unsqueeze(0L).unsqueeze (0L)
    input * cosines + rotateHalf input * sines

type SmolAttention = {
    Query: Linear
    Key: Linear
    Value: Linear
    Output: Linear
} with

    member this.forward(input: Tensor, cosines: Tensor, sines: Tensor) : Tensor =
        let batchSize = input.shape[0]
        let sequenceLength = input.shape[1]
        let headSize = hiddenSize / numHeads

        let toHeads heads (tensor: Tensor) =
            tensor.reshape([| batchSize; sequenceLength; heads; headSize |]).transpose (1L, 2L)

        let query = this.Query.forward input |> toHeads numHeads
        let key = this.Key.forward input |> toHeads numKeyValueHeads
        let value = this.Value.forward input |> toHeads numKeyValueHeads
        let query = applyRotaryEmbedding query cosines sines
        let key = applyRotaryEmbedding key cosines sines
        let groups = numHeads / numKeyValueHeads
        let key = key.repeat_interleave (groups, dim = 1L)
        let value = value.repeat_interleave (groups, dim = 1L)

        let attended =
            torch.nn.functional.scaled_dot_product_attention (query, key, value, is_casual = true)

        attended.transpose(1L, 2L).contiguous().reshape ([| batchSize; sequenceLength; hiddenSize |])
        |> this.Output.forward

type SmolMlp = {
    Gate: Linear
    Up: Linear
    Down: Linear
} with

    member this.forward(input: Tensor) : Tensor =
        this.Down.forward (this.Gate.forward(input).silu () * this.Up.forward input)

type SmolBlock = {
    InputNorm: RmsNorm
    Attention: SmolAttention
    PostAttentionNorm: RmsNorm
    Mlp: SmolMlp
} with

    member this.forward(input: Tensor, cosines: Tensor, sines: Tensor) : Tensor =
        let attended =
            this.InputNorm.forward input
            |> fun hidden -> this.Attention.forward (hidden, cosines, sines)

        let hidden = input + attended
        let projected = hidden |> this.PostAttentionNorm.forward |> this.Mlp.forward
        hidden + projected

type SmolLm2 = {
    TokenEmbedding: Embedding
    Blocks: SmolBlock list
    FinalNorm: RmsNorm
} with

    member this.forward(tokens: Tensor) : Tensor =
        let sequenceLength = tokens.shape[1]

        if sequenceLength > maxPositions then
            invalidArg (nameof tokens) $"SmolLM2 accepts at most {maxPositions} tokens."

        let headSize = hiddenSize / numHeads
        let cosines, sines = rotaryEmbedding sequenceLength headSize modelDtype
        let hidden = this.TokenEmbedding.forward tokens

        let hidden =
            this.Blocks
            |> List.fold (fun state block -> block.forward (state, cosines, sines)) hidden

        let hidden = this.FinalNorm.forward hidden
        hidden.matmul (this.TokenEmbedding.Embeddings.t ())

let createNorm () =
    RmsNorm.init hiddenSize 1e-5 modelDtype torch.CPU

let createBlock () = {
    InputNorm = createNorm ()
    Attention = {
        Query = Linear.initNoBias hiddenSize hiddenSize modelDtype torch.CPU
        Key = Linear.initNoBias hiddenSize (numKeyValueHeads * hiddenSize / numHeads) modelDtype torch.CPU
        Value = Linear.initNoBias hiddenSize (numKeyValueHeads * hiddenSize / numHeads) modelDtype torch.CPU
        Output = Linear.initNoBias hiddenSize hiddenSize modelDtype torch.CPU
    }
    PostAttentionNorm = createNorm ()
    Mlp = {
        Gate = Linear.initNoBias hiddenSize intermediateSize modelDtype torch.CPU
        Up = Linear.initNoBias hiddenSize intermediateSize modelDtype torch.CPU
        Down = Linear.initNoBias intermediateSize hiddenSize modelDtype torch.CPU
    }
}

let createModel () = {
    TokenEmbedding = Embedding.init vocabSize hiddenSize modelDtype torch.CPU
    Blocks = List.init numLayers (fun _ -> createBlock ())
    FinalNorm = createNorm ()
}

let nameMapping =
    let layer sourceSuffix targetSuffix =
        NameRule.rewrite ("model.layers.{layer}." + sourceSuffix) ("Blocks.{layer}." + targetSuffix)

    NameMapping.create [
        NameRule.rename "model.embed_tokens.weight" "TokenEmbedding.Embeddings"
        layer "input_layernorm.weight" "InputNorm.Inner.Weight"
        layer "self_attn.q_proj.weight" "Attention.Query.Weight"
        layer "self_attn.k_proj.weight" "Attention.Key.Weight"
        layer "self_attn.v_proj.weight" "Attention.Value.Weight"
        layer "self_attn.o_proj.weight" "Attention.Output.Weight"
        layer "post_attention_layernorm.weight" "PostAttentionNorm.Inner.Weight"
        layer "mlp.gate_proj.weight" "Mlp.Gate.Weight"
        layer "mlp.up_proj.weight" "Mlp.Up.Weight"
        layer "mlp.down_proj.weight" "Mlp.Down.Weight"
        NameRule.rename "model.norm.weight" "FinalNorm.Inner.Weight"
    ]

let hubFile filename = {
    RepoId = repoId
    Revision = revision
    Filename = filename
}

let loadModel () =
    let model = createModel ()

    let weightPath =
        Hub.download (hubFile "model.safetensors")
        |> Async.RunSynchronously

    use reader = SafeTensors.openFile weightPath

    let report =
        ModelState.loadSafeTensorsWith nameMapping Strict (Model.state model) reader

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
            SpecialTokens = [ "<|endoftext|>", 0; "<|im_start|>", 1; "<|im_end|>", 2 ]
            UnknownToken = Some "<|endoftext|>"
            PreTokenizer = ByteLevelPreTokenizer
    }

let chatPrompt userPrompt =
    $"<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n<|im_start|>user\n{userPrompt}<|im_end|>\n<|im_start|>assistant\n"

let generate (model: SmolLm2) (tokenizer: Tokenizer) maxNewTokens userPrompt =
    let promptTokens = tokenizer.encode (chatPrompt userPrompt) |> List.map int64
    let context = ResizeArray<int64>(promptTokens)
    let response = ResizeArray<int>()

    if context.Count = 0 then
        invalidArg (nameof userPrompt) "The prompt must produce at least one token."

    if int64 context.Count + int64 maxNewTokens > maxPositions then
        invalidArg (nameof maxNewTokens) $"The prompt and generated text must fit within {maxPositions} tokens."

    let mutable finished = false

    for _ in 1..maxNewTokens do
        if not finished then
            let nextToken =
                Toro.inferenceMode (fun () ->
                    scoped {
                        let input =
                            torch.tensor (context.ToArray(), dtype = torch.int64, device = torch.CPU)

                        let logits = model.forward (input.unsqueeze (0L))
                        return (logits.at [ I 0; I -1 ]).argmax(0L).ToInt64()
                    })

            if nextToken = eosTokenId then
                finished <- true
            else
                context.Add nextToken
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
    let model, report = loadModel ()
    printfn "Loaded %d tensors." report.Loaded.Length
    printfn "Prompt: %s" prompt
    printfn ""
    printfn "%s" (generate model tokenizer maxNewTokens prompt)
    0
