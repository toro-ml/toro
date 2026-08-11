// Load DistilBERT for sentiment analysis from Hugging Face Hub.
//
// Usage:
//   dotnet run                                                         Default repo
//   dotnet run -- distilbert/distilbert-base-uncased-finetuned-sst-2-english

open TorchSharp
open Toro
open Toro.NN
open Toro.Hub
open Toro.Text

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
    let q = Linear.init dim dim torch.float32 torch.CPU
    let k = Linear.init dim dim torch.float32 torch.CPU
    let v = Linear.init dim dim torch.float32 torch.CPU
    let outLin = Linear.init dim dim torch.float32 torch.CPU

    {
        Q = q
        K = k
        V = v
        OutLin = outLin
        NumHeads = numHeads
        HeadDim = headDim
    }

let createLayer () =
    let attn = createAttention ()
    let saNorm = LayerNorm.initDefault dim torch.float32 torch.CPU
    let ffn1 = Linear.init dim ffDim torch.float32 torch.CPU
    let ffn2 = Linear.init ffDim dim torch.float32 torch.CPU
    let outputNorm = LayerNorm.initDefault dim torch.float32 torch.CPU

    {
        Attention = attn
        SaNorm = saNorm
        Ffn1 = ffn1
        Ffn2 = ffn2
        OutputNorm = outputNorm
    }

let createModel () =
    let wordEmb = Embedding.init vocabSize dim torch.float32 torch.CPU
    let posEmb = Embedding.init maxPositions dim torch.float32 torch.CPU
    let embNorm = LayerNorm.initDefault dim torch.float32 torch.CPU

    let layers = [ for _ in 0 .. nLayers - 1 -> createLayer () ]

    let preClassifier = Linear.init dim dim torch.float32 torch.CPU
    let classifier = Linear.init dim 2 torch.float32 torch.CPU

    {
        WordEmbeddings = wordEmb
        PositionEmbeddings = posEmb
        EmbNorm = embNorm
        Layers = layers
        PreClassifier = preClassifier
        Classifier = classifier
    }

// ---------------------------------------------------------------------------
// Forward pass
// ---------------------------------------------------------------------------

let forwardAttention (attn: DistilBertAttention) (hidden: Tensor) =
    let batchSz = int hidden.shape[0]
    let seqLen = int hidden.shape[1]

    let q = attn.Q.forward hidden
    let k = attn.K.forward hidden
    let v = attn.V.forward hidden

    let q =
        q.reshape ([| int64 batchSz; int64 seqLen; int64 attn.NumHeads; int64 attn.HeadDim |])

    let q = q.permute ([| 0L; 2L; 1L; 3L |])

    let k =
        k.reshape ([| int64 batchSz; int64 seqLen; int64 attn.NumHeads; int64 attn.HeadDim |])

    let k = k.permute ([| 0L; 2L; 1L; 3L |])

    let v =
        v.reshape ([| int64 batchSz; int64 seqLen; int64 attn.NumHeads; int64 attn.HeadDim |])

    let v = v.permute ([| 0L; 2L; 1L; 3L |])

    let a = torch.nn.functional.scaled_dot_product_attention (q, k, v)
    let a = a.permute ([| 0L; 2L; 1L; 3L |])
    let a = a.contiguous ()

    let a =
        a.reshape ([| int64 batchSz; int64 seqLen; int64 (attn.NumHeads * attn.HeadDim) |])

    attn.OutLin.forward a

let forwardLayer (layer: DistilBertLayer) (hidden: Tensor) =
    // Self-attention with post-norm
    let attnOut = forwardAttention layer.Attention hidden
    let hidden = hidden.add attnOut
    let hidden = layer.SaNorm.forward hidden

    // FFN with post-norm
    let ffnOut = layer.Ffn1.forward hidden
    let ffnOut = ffnOut.gelu ()
    let ffnOut = layer.Ffn2.forward ffnOut
    let hidden = hidden.add ffnOut
    layer.OutputNorm.forward hidden

let forward (model: DistilBertClassifier) (inputIds: Tensor) =
    let seqLen = inputIds.shape[1]

    // Embeddings: word + position
    let posIds =
        torch.arange (seqLen, dtype = torch.int64, device = torch.CPU)
        |> fun t -> t.unsqueeze (0L)

    let wordEmb = model.WordEmbeddings.forward inputIds
    let posEmb = model.PositionEmbeddings.forward posIds
    let hidden = wordEmb.add posEmb
    let hidden = model.EmbNorm.forward hidden

    // Transformer layers
    let hidden =
        model.Layers
        |> List.fold (fun h layer -> forwardLayer layer h) hidden

    // Classifier: [CLS] token → pre-classifier → ReLU → classifier
    let cls = hidden.at [ A; I 0 ]
    let cls = model.PreClassifier.forward cls
    let cls = cls.relu ()
    model.Classifier.forward cls

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
// Tokenizer (WordPiece via Toro.Text)
// ---------------------------------------------------------------------------

let loadTokenizer (repoId: string) : Async<Tokenizer> =
    async {
        let! path = Hub.download repoId "vocab.txt"

        return
            Tokenizer.fromWordPiece {
                WordPieceConfig.create path with
                    SpecialTokens = [ "[UNK]", 100; "[CLS]", 101; "[SEP]", 102; "[PAD]", 0 ]
                    PreTokenizer = Regex @"\w+|[^\w\s]+"
                    Normalizer = LowerCase
            }
    }

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

    printfn "Downloading %s ..." repoId

    let weights =
        Hub.loadSafeTensors repoId "model.safetensors"
        |> Async.RunSynchronously

    let tokenizer = loadTokenizer repoId |> Async.RunSynchronously

    printfn "Downloaded %d tensors, tokenizer ready" (weights: Map<string, Tensor>).Count

    let model = createModel ()
    let nameMap = buildNameMap ()
    let report = Model.loadFromDict model weights (Some nameMap) Lenient
    printfn "Model loaded (%d params, %d skipped)" report.Loaded.Length report.Missing.Length
    printfn ""

    for text in testTexts do
        let logits =
            Toro.noGrad (fun () ->
                let ids = tokenizer.encode text
                let withSpecial = 101 :: ids @ [ 102 ]
                let data = withSpecial |> List.map int64 |> List.toArray

                let input =
                    torch.tensor (data, dtype = torch.int64, device = torch.CPU)
                    |> fun t -> t.unsqueeze (int64 0)

                forward model input)

        let pred = logits.argmax (int64 1)
        let idx = pred[0].ToSingle()
        let label = labels[int idx]
        printfn "  %-25s -> %s" text label

    0
