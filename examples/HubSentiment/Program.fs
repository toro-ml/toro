// Load DistilBERT for sentiment analysis from Hugging Face Hub.
//
// Usage:
//   dotnet run                                                         Default repo
//   dotnet run -- distilbert/distilbert-base-uncased-finetuned-sst-2-english

open System
open System.IO
open TorchSharp
open Toro
open Toro.NN
open Toro.Hub

// ---------------------------------------------------------------------------
// DistilBERT model definition
// ---------------------------------------------------------------------------
// DistilBERT uses post-norm (Attn → Add → Norm) unlike Toro.NN.TransformerBlock
// which uses pre-norm. Custom record types implement the exact architecture so
// that safetensors weights from Hugging Face load without modification.

type DistilBertAttention = {
    Q: Linear
    K: Linear
    V: Linear
    OutLin: Linear
    NumHeads: int
    HeadDim: int
}

type DistilBertLayer = {
    Attention: DistilBertAttention
    SaNorm: LayerNorm
    Ffn1: Linear
    Ffn2: Linear
    OutputNorm: LayerNorm
}

type DistilBertClassifier = {
    WordEmbeddings: Embedding
    PositionEmbeddings: Embedding
    EmbNorm: LayerNorm
    Layers: DistilBertLayer list
    PreClassifier: Linear
    Classifier: Linear
}

// DistilBERT config
let vocabSize = 30522
let maxPositions = 512
let dim = 768
let numHeads = 12
let headDim = dim / numHeads
let ffDim = 3072
let nLayers = 6

// ---------------------------------------------------------------------------
// Model construction
// ---------------------------------------------------------------------------

let createAttention () =
    result {
        let! q = Linear.init dim dim F32 Cpu
        let! k = Linear.init dim dim F32 Cpu
        let! v = Linear.init dim dim F32 Cpu
        let! outLin = Linear.init dim dim F32 Cpu

        return {
            Q = q
            K = k
            V = v
            OutLin = outLin
            NumHeads = numHeads
            HeadDim = headDim
        }
    }

let createLayer () =
    result {
        let! attn = createAttention ()
        let! saNorm = LayerNorm.initDefault dim F32 Cpu
        let! ffn1 = Linear.init dim ffDim F32 Cpu
        let! ffn2 = Linear.init ffDim dim F32 Cpu
        let! outputNorm = LayerNorm.initDefault dim F32 Cpu

        return {
            Attention = attn
            SaNorm = saNorm
            Ffn1 = ffn1
            Ffn2 = ffn2
            OutputNorm = outputNorm
        }
    }

let createModel () =
    result {
        let! wordEmb = Embedding.init vocabSize dim F32 Cpu
        let! posEmb = Embedding.init maxPositions dim F32 Cpu
        let! embNorm = LayerNorm.initDefault dim F32 Cpu

        let! layers =
            [ for _ in 0 .. nLayers - 1 -> createLayer () ]
            |> List.fold
                (fun acc r ->
                    result {
                        let! lst = acc
                        let! layer = r
                        return lst @ [ layer ]
                    })
                (Ok [])

        let! preClassifier = Linear.init dim dim F32 Cpu
        let! classifier = Linear.init dim 2 F32 Cpu

        return {
            WordEmbeddings = wordEmb
            PositionEmbeddings = posEmb
            EmbNorm = embNorm
            Layers = layers
            PreClassifier = preClassifier
            Classifier = classifier
        }
    }

// ---------------------------------------------------------------------------
// Forward pass
// ---------------------------------------------------------------------------

let forwardAttention (attn: DistilBertAttention) (hidden: Tensor) =
    result {
        let batchSz = hidden.Shape[0]
        let seqLen = hidden.Shape[1]

        let! q = attn.Q.forward hidden
        let! k = attn.K.forward hidden
        let! v = attn.V.forward hidden

        let! q = q.reshape [ batchSz; seqLen; attn.NumHeads; attn.HeadDim ]
        let! q = q.permute [ 0; 2; 1; 3 ]
        let! k = k.reshape [ batchSz; seqLen; attn.NumHeads; attn.HeadDim ]
        let! k = k.permute [ 0; 2; 1; 3 ]
        let! v = v.reshape [ batchSz; seqLen; attn.NumHeads; attn.HeadDim ]
        let! v = v.permute [ 0; 2; 1; 3 ]

        let! a = q.scaledDotProductAttention (k, v)
        let! a = a.permute [ 0; 2; 1; 3 ]
        let! a = a.contiguous ()
        let! a = a.reshape [ batchSz; seqLen; attn.NumHeads * attn.HeadDim ]
        return! attn.OutLin.forward a
    }

let forwardLayer (layer: DistilBertLayer) (hidden: Tensor) =
    result {
        // Self-attention with post-norm
        let! attnOut = forwardAttention layer.Attention hidden
        let! hidden = hidden.add attnOut
        let! hidden = layer.SaNorm.forward hidden

        // FFN with post-norm
        let! ffnOut = layer.Ffn1.forward hidden
        let! ffnOut = ffnOut.gelu ()
        let! ffnOut = layer.Ffn2.forward ffnOut
        let! hidden = hidden.add ffnOut
        return! layer.OutputNorm.forward hidden
    }

let forward (model: DistilBertClassifier) (inputIds: Tensor) =
    result {
        let seqLen = inputIds.Shape[1]

        // Embeddings: word + position
        let! posIds = Tensor.arange (float seqLen, I64, Cpu)
        let! posIds = posIds.unsqueeze 0
        let! wordEmb = model.WordEmbeddings.forward inputIds
        let! posEmb = model.PositionEmbeddings.forward posIds
        let! hidden = wordEmb.add posEmb
        let! hidden = model.EmbNorm.forward hidden

        // Transformer layers
        let! hidden =
            (Ok hidden, model.Layers)
            ||> List.fold (fun acc layer ->
                result {
                    let! h = acc
                    return! forwardLayer layer h
                })

        // Classifier: [CLS] token → pre-classifier → ReLU → classifier
        let cls = hidden.at [ A; I 0 ]
        let! cls = model.PreClassifier.forward cls
        let! cls = cls.relu ()
        return! model.Classifier.forward cls
    }

// ---------------------------------------------------------------------------
// HF name mapping
// ---------------------------------------------------------------------------

let buildNameMap () =
    let emb = [
        "distilbert.embeddings.word_embeddings.weight", "WordEmbeddings.Embeddings"
        "distilbert.embeddings.position_embeddings.weight", "PositionEmbeddings.Embeddings"
        "distilbert.embeddings.LayerNorm.weight", "EmbNorm.Weight"
        "distilbert.embeddings.LayerNorm.bias", "EmbNorm.Bias"
    ]

    let layers = [
        for i in 0 .. nLayers - 1 do
            let hf = $"distilbert.transformer.layer.{i}"
            let toro = $"Layers.{i}"
            yield $"{hf}.attention.q_lin.weight", $"{toro}.Attention.Q.Weight"
            yield $"{hf}.attention.q_lin.bias", $"{toro}.Attention.Q.Bias"
            yield $"{hf}.attention.k_lin.weight", $"{toro}.Attention.K.Weight"
            yield $"{hf}.attention.k_lin.bias", $"{toro}.Attention.K.Bias"
            yield $"{hf}.attention.v_lin.weight", $"{toro}.Attention.V.Weight"
            yield $"{hf}.attention.v_lin.bias", $"{toro}.Attention.V.Bias"
            yield $"{hf}.attention.out_lin.weight", $"{toro}.Attention.OutLin.Weight"
            yield $"{hf}.attention.out_lin.bias", $"{toro}.Attention.OutLin.Bias"
            yield $"{hf}.sa_layer_norm.weight", $"{toro}.SaNorm.Weight"
            yield $"{hf}.sa_layer_norm.bias", $"{toro}.SaNorm.Bias"
            yield $"{hf}.ffn.lin1.weight", $"{toro}.Ffn1.Weight"
            yield $"{hf}.ffn.lin1.bias", $"{toro}.Ffn1.Bias"
            yield $"{hf}.ffn.lin2.weight", $"{toro}.Ffn2.Weight"
            yield $"{hf}.ffn.lin2.bias", $"{toro}.Ffn2.Bias"
            yield $"{hf}.output_layer_norm.weight", $"{toro}.OutputNorm.Weight"
            yield $"{hf}.output_layer_norm.bias", $"{toro}.OutputNorm.Bias"
    ]

    let head = [
        "pre_classifier.weight", "PreClassifier.Weight"
        "pre_classifier.bias", "PreClassifier.Bias"
        "classifier.weight", "Classifier.Weight"
        "classifier.bias", "Classifier.Bias"
    ]

    Map(emb @ layers @ head)

// ---------------------------------------------------------------------------
// WordPiece tokenizer
// ---------------------------------------------------------------------------

let loadVocab (repoId: string) : Async<Result<Map<string, int>, ToroError>> =
    async {
        let! pathResult = Hub.download repoId "vocab.txt"

        return
            pathResult
            |> Result.map (fun path ->
                File.ReadAllLines(path)
                |> Array.mapi (fun i token -> token, i)
                |> Map.ofArray)
    }

/// Greedy longest-match WordPiece tokenization for a single word.
let wordPiece (vocab: Map<string, int>) (word: string) : int list =
    let unkId = vocab["[UNK]"]

    let rec findLongest start endPos =
        if endPos <= start then
            None
        else
            let sub = word[start .. endPos - 1]
            let candidate = if start > 0 then "##" + sub else sub

            match vocab |> Map.tryFind candidate with
            | Some id -> Some(id, endPos)
            | None -> findLongest start (endPos - 1)

    let rec loop start acc =
        if start >= word.Length then
            Some(List.rev acc)
        else
            match findLongest start word.Length with
            | Some(id, next) -> loop next (id :: acc)
            | None -> None

    loop 0 [] |> Option.defaultValue [ unkId ]

/// Tokenize text with [CLS] and [SEP].
let tokenize (vocab: Map<string, int>) (text: string) : int64 array =
    let words =
        text.ToLowerInvariant().Split([| ' '; '\t' |], StringSplitOptions.RemoveEmptyEntries)

    [|
        yield int64 vocab["[CLS]"]

        for w in words do
            yield! wordPiece vocab w |> List.map int64

        yield int64 vocab["[SEP]"]
    |]

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

let testTexts = [|
    "this movie is great"
    "i love it so much"
    "what a wonderful day"
    "this is terrible"
    "worst film ever made"
    "i hate this movie"
|]

let labels = [| "NEGATIVE"; "POSITIVE" |]

[<EntryPoint>]
let main argv =
    let repoId =
        Array.tryItem 0 argv
        |> Option.defaultValue "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

    result {
        printfn "Downloading %s ..." repoId

        let! weights =
            Hub.loadSafeTensors repoId "model.safetensors"
            |> Async.RunSynchronously

        let! vocab = loadVocab repoId |> Async.RunSynchronously
        printfn "Downloaded %d tensors, vocab %d tokens" weights.Count vocab.Count

        let! model = createModel ()
        let nameMap = buildNameMap ()
        do! Model.loadFromDict model weights (Some nameMap)
        printfn "Model loaded"
        printfn ""

        for text in testTexts do
            let tokenIds = tokenize vocab text

            let! logits =
                Toro.noGrad (fun () ->
                    result {
                        let! input = Tensor.ofTorchTensor (torch.tensor (tokenIds, dtype = torch.int64))
                        let! input = input.unsqueeze 0
                        return! forward model input
                    })

            let! pred = logits.argmax 1
            let! idx = pred[0].toFloat32Scalar ()
            let label = labels[int idx]
            printfn "  %-25s -> %s" text label
    }
    |> function
        | Ok() -> 0
        | Error e ->
            eprintfn "%A" e
            1
