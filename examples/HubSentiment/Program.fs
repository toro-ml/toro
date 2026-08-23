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

// DistilBERT uses post-norm blocks. Embeddings and the classification head remain
// model-specific so Hugging Face safetensors load without a prefix rewrite.

type DistilBertClassifier = {
    WordEmbeddings: Embedding
    PositionEmbeddings: Embedding
    EmbNorm: LayerNorm
    Layers: PostNormTransformerBlock list
    PreClassifier: Linear
    Classifier: Linear
}

let vocabSize = 30522L
let maxPositions = 512L
let dim = 768L
let numHeads = 12L
let ffDim = 3072L
let nLayers = 6

let createModel () =
    let wordEmb = Embedding.init vocabSize dim torch.float32 torch.CPU
    let posEmb = Embedding.init maxPositions dim torch.float32 torch.CPU
    let embNorm = LayerNorm.initDefault dim torch.float32 torch.CPU

    let layers = [
        for _ in 0 .. nLayers - 1 -> PostNormTransformerBlock.initDefault dim numHeads ffDim torch.float32 torch.CPU
    ]

    let preClassifier = Linear.init dim dim torch.float32 torch.CPU
    let classifier = Linear.init dim 2L torch.float32 torch.CPU

    {
        WordEmbeddings = wordEmb
        PositionEmbeddings = posEmb
        EmbNorm = embNorm
        Layers = layers
        PreClassifier = preClassifier
        Classifier = classifier
    }

let forward (model: DistilBertClassifier) (inputIds: Tensor) =
    let seqLen = inputIds.shape[1]

    let posIds =
        torch.arange (seqLen, dtype = torch.int64, device = torch.CPU)
        |> fun t -> t.unsqueeze 0L

    let hidden =
        model.WordEmbeddings.forward inputIds
        |> fun wordEmb -> wordEmb.add (model.PositionEmbeddings.forward posIds)
        |> model.EmbNorm.forward

    let hidden =
        model.Layers
        |> List.fold (fun h layer -> layer.forward h) hidden

    hidden.at [ A; I 0 ]
    |> model.PreClassifier.forward
    |> fun cls -> cls.relu ()
    |> model.Classifier.forward

let nameMapping =
    let layer sourceSuffix targetSuffix =
        NameRule.rewrite ("distilbert.transformer.layer.{layer}." + sourceSuffix) ("Layers.{layer}." + targetSuffix)

    NameMapping.create [
        NameRule.rename "distilbert.embeddings.word_embeddings.weight" "WordEmbeddings.Embeddings"
        NameRule.rename "distilbert.embeddings.position_embeddings.weight" "PositionEmbeddings.Embeddings"
        NameRule.rename "distilbert.embeddings.LayerNorm.weight" "EmbNorm.Weight"
        NameRule.rename "distilbert.embeddings.LayerNorm.bias" "EmbNorm.Bias"
        layer "attention.q_lin.weight" "Attn.WQ.Weight"
        layer "attention.q_lin.bias" "Attn.WQ.Bias"
        layer "attention.k_lin.weight" "Attn.WK.Weight"
        layer "attention.k_lin.bias" "Attn.WK.Bias"
        layer "attention.v_lin.weight" "Attn.WV.Weight"
        layer "attention.v_lin.bias" "Attn.WV.Bias"
        layer "attention.out_lin.weight" "Attn.WO.Weight"
        layer "attention.out_lin.bias" "Attn.WO.Bias"
        layer "sa_layer_norm.weight" "AttnNorm.Weight"
        layer "sa_layer_norm.bias" "AttnNorm.Bias"
        layer "ffn.lin1.weight" "Ff1.Weight"
        layer "ffn.lin1.bias" "Ff1.Bias"
        layer "ffn.lin2.weight" "Ff2.Weight"
        layer "ffn.lin2.bias" "Ff2.Bias"
        layer "output_layer_norm.weight" "FfNorm.Weight"
        layer "output_layer_norm.bias" "FfNorm.Bias"
        NameRule.rename "pre_classifier.weight" "PreClassifier.Weight"
        NameRule.rename "pre_classifier.bias" "PreClassifier.Bias"
        NameRule.rename "classifier.weight" "Classifier.Weight"
        NameRule.rename "classifier.bias" "Classifier.Bias"
    ]

let loadTokenizer (repoId: string) (revision: string) : Async<Tokenizer> =
    async {
        let! path =
            Hub.download {
                RepoId = repoId
                Revision = revision
                Filename = "vocab.txt"
            }

        return Tokenizer.fromBert (BertConfig.create path)
    }

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

    let revision =
        Array.tryItem 1 argv
        |> Option.defaultValue "714eb0fa89d2f80546fda750413ed43d93601a13"

    printfn "Downloading %s at %s ..." repoId revision

    let weightPath =
        Hub.download {
            RepoId = repoId
            Revision = revision
            Filename = "model.safetensors"
        }
        |> Async.RunSynchronously

    let tokenizer = loadTokenizer repoId revision |> Async.RunSynchronously

    let model = createModel ()
    use reader = SafeTensors.openFile weightPath
    printfn "Downloaded %d tensors, tokenizer ready" reader.Metadata.Count

    let report =
        ModelState.loadSafeTensorsWith nameMapping Lenient (Model.state model) reader

    printfn "Model loaded (%d params, %d skipped)" report.Loaded.Length report.Missing.Length
    printfn ""

    for text in testTexts do
        let logits =
            Toro.noGrad (fun () ->
                let data = tokenizer.encode text |> List.toArray

                let input =
                    torch.tensor (data, dtype = torch.int64, device = torch.CPU)
                    |> fun t -> t.unsqueeze (int64 0)

                forward model input)

        let pred = logits.argmax (int64 1)
        let idx = pred[0].ToSingle()
        let label = labels[int idx]
        printfn "  %-25s -> %s" text label

    0
