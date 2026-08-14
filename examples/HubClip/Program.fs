open System
open System.Net.Http
open TorchSharp
open Toro
open Toro.Hub
open Toro.NN
open Toro.Text
open Toro.Vision

let repoId = "openai/clip-vit-base-patch32"
let revision = "c7244be81152024ce0e99ac8d2e373a8953d9f9a"

let textHiddenSize = 512L
let textIntermediateSize = 2048L
let textHeads = 8L
let textLayers = 12
let maxPositions = 77L
let vocabSize = 49408L

let visionHiddenSize = 768L
let visionIntermediateSize = 3072L
let visionHeads = 12L
let visionLayers = 12
let imageSize = 224L
let patchSize = 32L
let visionPositions = 50L

let projectionSize = 512L
let bosTokenId = 49406L
let eosTokenId = 49407L

type ClipAttention = {
    Query: Linear
    Key: Linear
    Value: Linear
    Output: Linear
    NumHeads: int64
} with

    member this.forward (causal: bool) (input: Tensor) : Tensor =
        let batchSize = input.shape[0]
        let sequenceLength = input.shape[1]
        let hiddenSize = input.shape[2]
        let headSize = hiddenSize / this.NumHeads

        let toHeads (projection: Linear) =
            projection.forward input
            |> _.reshape([| batchSize; sequenceLength; this.NumHeads; headSize |])
            |> _.permute([| 0L; 2L; 1L; 3L |])

        let query = toHeads this.Query
        let key = toHeads this.Key
        let value = toHeads this.Value

        let attended =
            torch.nn.functional.scaled_dot_product_attention (query, key, value, is_casual = causal)

        attended.permute([| 0L; 2L; 1L; 3L |]).contiguous().reshape ([| batchSize; sequenceLength; hiddenSize |])
        |> this.Output.forward

type ClipMlp = {
    Input: Linear
    Output: Linear
} with

    member this.forward(input: Tensor) : Tensor =
        let hidden = this.Input.forward input
        let activated = hidden * (hidden * scalar 1.702).sigmoid ()
        this.Output.forward activated

type ClipLayer = {
    Norm1: LayerNorm
    Attention: ClipAttention
    Norm2: LayerNorm
    Mlp: ClipMlp
} with

    member this.forward (causal: bool) (input: Tensor) : Tensor =
        let attended = input |> this.Norm1.forward |> this.Attention.forward causal
        let hidden = input + attended
        hidden + (hidden |> this.Norm2.forward |> this.Mlp.forward)

type TextEncoder = {
    TokenEmbedding: Embedding
    PositionEmbedding: Embedding
    Layers: ClipLayer list
    FinalNorm: LayerNorm
} with

    member this.forward(tokens: Tensor) : Tensor =
        let sequenceLength = tokens.shape[1]

        if sequenceLength > maxPositions then
            invalidArg (nameof tokens) $"CLIP accepts at most {maxPositions} text tokens."

        let positions =
            torch.arange (sequenceLength, dtype = torch.int64, device = torch.CPU)
            |> _.unsqueeze(0L)

        let hidden =
            this.TokenEmbedding.forward tokens
            + this.PositionEmbedding.forward positions

        let hidden =
            this.Layers
            |> List.fold (fun state layer -> layer.forward true state) hidden

        this.FinalNorm.forward hidden

type VisionEncoder = {
    [<Parameter>]
    ClassEmbedding: Tensor
    PatchEmbedding: Conv2d
    PositionEmbedding: Embedding
    PreNorm: LayerNorm
    Layers: ClipLayer list
    PostNorm: LayerNorm
} with

    member this.forward(input: Tensor) : Tensor =
        let batchSize = input.shape[0]

        let patches =
            this.PatchEmbedding.forward input
            |> _.flatten(2L, -1L)
            |> _.transpose(1L, 2L)

        let classTokens =
            this.ClassEmbedding.unsqueeze(0L).unsqueeze(0L).expand ([| batchSize; 1L; visionHiddenSize |])

        let hidden =
            torch.cat ([| classTokens; patches |], 1L)
            + this.PositionEmbedding.Embeddings.unsqueeze (0L)
            |> this.PreNorm.forward

        let hidden =
            this.Layers
            |> List.fold (fun state layer -> layer.forward false state) hidden

        hidden.at [ A; I 0 ] |> this.PostNorm.forward

type ClipModel = {
    Text: TextEncoder
    Vision: VisionEncoder
    TextProjection: Linear
    VisionProjection: Linear
    [<Parameter>]
    LogitScale: Tensor
} with

    member private _.normalize(features: Tensor) =
        features / features.square().sum(-1L, true).sqrt ()

    member this.encodeText(tokens: Tensor) : Tensor =
        let hidden = this.Text.forward tokens
        let pooled = hidden.at [ I 0; I -1 ] |> _.unsqueeze(0L)
        this.TextProjection.forward pooled |> this.normalize

    member this.encodeImage(input: Tensor) : Tensor =
        this.Vision.forward input
        |> this.VisionProjection.forward
        |> this.normalize

    member this.forward (image: Tensor) (texts: Tensor list) : Tensor =
        let imageFeatures = this.encodeImage image

        let textFeatures =
            texts
            |> List.map this.encodeText
            |> List.toArray
            |> torch.cat

        imageFeatures.matmul (textFeatures.t ())
        * this.LogitScale.exp ()

let createLayerNorm size =
    LayerNorm.init
        size
        {
            LayerNormConfig.defaultConfig with
                Eps = 1e-5
        }
        torch.float32
        torch.CPU

let createAttention hiddenSize numHeads = {
    Query = Linear.init hiddenSize hiddenSize torch.float32 torch.CPU
    Key = Linear.init hiddenSize hiddenSize torch.float32 torch.CPU
    Value = Linear.init hiddenSize hiddenSize torch.float32 torch.CPU
    Output = Linear.init hiddenSize hiddenSize torch.float32 torch.CPU
    NumHeads = numHeads
}

let createLayer hiddenSize intermediateSize numHeads = {
    Norm1 = createLayerNorm hiddenSize
    Attention = createAttention hiddenSize numHeads
    Norm2 = createLayerNorm hiddenSize
    Mlp = {
        Input = Linear.init hiddenSize intermediateSize torch.float32 torch.CPU
        Output = Linear.init intermediateSize hiddenSize torch.float32 torch.CPU
    }
}

let createModel () = {
    Text = {
        TokenEmbedding = Embedding.init vocabSize textHiddenSize torch.float32 torch.CPU
        PositionEmbedding = Embedding.init maxPositions textHiddenSize torch.float32 torch.CPU
        Layers = List.init textLayers (fun _ -> createLayer textHiddenSize textIntermediateSize textHeads)
        FinalNorm = createLayerNorm textHiddenSize
    }
    Vision = {
        ClassEmbedding = Init.toParam [| visionHiddenSize |] torch.float32 torch.CPU (Init.Const 0.0)
        PatchEmbedding =
            Conv2d.initNoBias
                3L
                visionHiddenSize
                patchSize
                {
                    Conv2dConfig.defaultConfig with
                        Stride = patchSize
                }
                torch.float32
                torch.CPU
        PositionEmbedding = Embedding.init visionPositions visionHiddenSize torch.float32 torch.CPU
        PreNorm = createLayerNorm visionHiddenSize
        Layers = List.init visionLayers (fun _ -> createLayer visionHiddenSize visionIntermediateSize visionHeads)
        PostNorm = createLayerNorm visionHiddenSize
    }
    TextProjection = Linear.initNoBias textHiddenSize projectionSize torch.float32 torch.CPU
    VisionProjection = Linear.initNoBias visionHiddenSize projectionSize torch.float32 torch.CPU
    LogitScale = Init.toParam [||] torch.float32 torch.CPU (Init.Const 2.6592)
}

let nameMapping =
    let projection sourcePrefix targetPrefix = [
        NameRule.rewrite (sourcePrefix + ".weight") (targetPrefix + ".Weight")
        NameRule.rewrite (sourcePrefix + ".bias") (targetPrefix + ".Bias")
    ]

    let layer sourcePrefix targetPrefix = [
        yield! projection (sourcePrefix + ".self_attn.q_proj") (targetPrefix + ".Attention.Query")
        yield! projection (sourcePrefix + ".self_attn.k_proj") (targetPrefix + ".Attention.Key")
        yield! projection (sourcePrefix + ".self_attn.v_proj") (targetPrefix + ".Attention.Value")
        yield! projection (sourcePrefix + ".self_attn.out_proj") (targetPrefix + ".Attention.Output")
        yield! projection (sourcePrefix + ".layer_norm1") (targetPrefix + ".Norm1")
        yield! projection (sourcePrefix + ".layer_norm2") (targetPrefix + ".Norm2")
        yield! projection (sourcePrefix + ".mlp.fc1") (targetPrefix + ".Mlp.Input")
        yield! projection (sourcePrefix + ".mlp.fc2") (targetPrefix + ".Mlp.Output")
    ]

    NameMapping.create [
        NameRule.ignoreSuffix "position_ids"
        NameRule.rename "logit_scale" "LogitScale"
        NameRule.rename "text_projection.weight" "TextProjection.Weight"
        NameRule.rename "visual_projection.weight" "VisionProjection.Weight"
        NameRule.rename "text_model.embeddings.token_embedding.weight" "Text.TokenEmbedding.Embeddings"
        NameRule.rename "text_model.embeddings.position_embedding.weight" "Text.PositionEmbedding.Embeddings"
        yield! layer "text_model.encoder.layers.{layer}" "Text.Layers.{layer}"
        NameRule.rename "text_model.final_layer_norm.weight" "Text.FinalNorm.Weight"
        NameRule.rename "text_model.final_layer_norm.bias" "Text.FinalNorm.Bias"
        NameRule.rename "vision_model.embeddings.class_embedding" "Vision.ClassEmbedding"
        NameRule.rename "vision_model.embeddings.patch_embedding.weight" "Vision.PatchEmbedding.Weight"
        NameRule.rename "vision_model.embeddings.position_embedding.weight" "Vision.PositionEmbedding.Embeddings"
        NameRule.rename "vision_model.pre_layrnorm.weight" "Vision.PreNorm.Weight"
        NameRule.rename "vision_model.pre_layrnorm.bias" "Vision.PreNorm.Bias"
        yield! layer "vision_model.encoder.layers.{layer}" "Vision.Layers.{layer}"
        NameRule.rename "vision_model.post_layernorm.weight" "Vision.PostNorm.Weight"
        NameRule.rename "vision_model.post_layernorm.bias" "Vision.PostNorm.Bias"
    ]

let hubFile filename = {
    RepoId = repoId
    Revision = revision
    Filename = filename
}

let loadModel () =
    let model = createModel ()

    let weights =
        Hub.loadSafeTensors (hubFile "model.safetensors")
        |> Async.RunSynchronously

    let report =
        try
            weights |> Model.loadFromDictWith nameMapping Strict model
        finally
            for tensor in weights.Values do
                tensor.Dispose()

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
            SpecialTokens = [ "<|startoftext|>", int bosTokenId; "<|endoftext|>", int eosTokenId ]
            UnknownToken = Some "<|endoftext|>"
            EndOfWordSuffix = Some "</w>"
            PreTokenizer =
                Regex "<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+"
            Normalizer = LowerCase
    }

let encodePrompt (tokenizer: Tokenizer) label =
    let prompt = $"a photo of a {label}"
    let body = tokenizer.encode prompt |> List.map int64
    let tokens = bosTokenId :: body @ [ eosTokenId ]

    if int64 tokens.Length > maxPositions then
        invalidArg (nameof label) $"Prompt for label '{label}' exceeds {maxPositions} tokens."

    torch.tensor (List.toArray tokens, dtype = torch.int64, device = torch.CPU)
    |> _.unsqueeze(0L)

let loadImage source =
    match Uri.TryCreate(source, UriKind.Absolute) with
    | true, uri when
        uri.Scheme = Uri.UriSchemeHttp
        || uri.Scheme = Uri.UriSchemeHttps
        ->
        use client = new HttpClient()
        use stream = client.GetStreamAsync(uri).GetAwaiter().GetResult()
        Image.loadStream stream torch.CPU
    | _ -> Image.load source torch.CPU

let preprocess image =
    let resize =
        ResizeShortestEdge.create (int imageSize) torch.InterpolationMode.Bicubic

    let crop = CenterCrop.create (int imageSize) (int imageSize)

    let normalize: Normalize = {
        Mean = [ 0.48145466; 0.4578275; 0.40821073 ]
        Std = [ 0.26862954; 0.26130258; 0.27577711 ]
    }

    image
    |> resize.apply
    |> crop.apply
    |> normalize.apply
    |> _.unsqueeze(0L)

[<EntryPoint>]
let main argv =
    let defaultSource =
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/parrots.png"

    let source = Array.tryItem 0 argv |> Option.defaultValue defaultSource

    let labels =
        if argv.Length > 1 then
            argv[1..] |> Array.toList
        else
            [ "parrot"; "bird"; "dog" ]

    printfn "Loading %s at %s ..." repoId revision
    let tokenizer = loadTokenizer ()
    let model, report = loadModel ()
    printfn "Loaded %d tensors." report.Loaded.Length

    scoped {
        let image = loadImage source |> preprocess
        let texts = labels |> List.map (encodePrompt tokenizer)

        let probabilities =
            Toro.inferenceMode (fun () -> model.forward image texts |> _.softmax(-1L))

        let values = probabilities.flatten().data<float32> ()

        printfn "Zero-shot predictions for %s:" source

        labels
        |> List.mapi (fun index label -> label, float values[index])
        |> List.sortByDescending snd
        |> List.iteri (fun rank (label, probability) -> printfn "  %d. %-24s %.2f%%" (rank + 1) label (probability * 100.0))
    }

    0
