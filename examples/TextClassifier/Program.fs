open Toro
open Toro.NN

// Char-level sentiment classification with Transformer.
// positive (label 0) vs negative (label 1).

let positiveWords = [|
    "good"
    "great"
    "happy"
    "love"
    "wonderful"
    "amazing"
    "excellent"
    "fantastic"
    "brilliant"
    "superb"
    "nice"
    "fine"
    "perfect"
    "pleasant"
    "delightful"
|]

let negativeWords = [|
    "bad"
    "terrible"
    "sad"
    "hate"
    "awful"
    "horrible"
    "poor"
    "dreadful"
    "ugly"
    "nasty"
    "wrong"
    "worse"
    "worst"
    "grim"
    "miserable"
|]

let maxLen = 10

let allChars =
    Array.append positiveWords negativeWords
    |> Array.collect _.ToCharArray()
    |> Array.distinct
    |> Array.sort

let padChar = '\000'
let vocab = Array.append [| padChar |] allChars
let vocabSize = vocab.Length
let charToIdx = vocab |> Array.mapi (fun i c -> c, i) |> Map.ofArray

let encodeWord (w: string) =
    let padded = w.PadRight(maxLen, padChar)

    padded
    |> Seq.map (fun c -> int64 (charToIdx.TryFind c |> Option.defaultValue 0))
    |> Seq.toArray

type TransformerClassifier = {
    Embed: Embedding
    Block: TransformerBlock
    Head: Linear
}

let dim = 64
let numHeads = 4
let ffDim = 128

let createModel () =
    let embed = Embedding.init vocabSize dim F32 Cpu
    let block = TransformerBlock.init dim numHeads ffDim F32 Cpu
    let head = Linear.init dim 2 F32 Cpu

    {
        Embed = embed
        Block = block
        Head = head
    }

let forward (model: TransformerClassifier) (input: Tensor) =
    let x = model.Embed.forward input
    let x = model.Block.forward x
    let pooled = x.mean (1)
    model.Head.forward pooled

[<EntryPoint>]
let main _argv =
    let nPos = positiveWords.Length
    let nNeg = negativeWords.Length
    let nSamples = nPos + nNeg

    let inputData =
        Array.append positiveWords negativeWords
        |> Array.collect (fun w -> encodeWord w)

    let labelData = Array.append (Array.create nPos 0L) (Array.create nNeg 1L)

    let input = Tensor.ofArray (inputData, Cpu)
    let input = input.reshape [ nSamples; maxLen ]
    let labels = Tensor.ofArray (labelData, Cpu)

    printfn "Text classifier: positive vs negative words"
    printfn "Samples: %d (%d pos + %d neg), vocab: %d chars, maxLen: %d" nSamples nPos nNeg vocabSize maxLen

    printfn
        "Model: Embedding(%d,%d) -> TransformerBlock(heads=%d, ff=%d) -> mean pool -> Linear(%d,2)"
        vocabSize
        dim
        numHeads
        ffDim
        dim

    printfn ""

    let model = createModel ()
    let opt = AdamW.createWithLr 1e-3 (Model.trainableVars model)

    for epoch in 1..100 do
        scoped {
            opt.zeroGrad ()
            let logits = forward model input
            let loss = Loss.crossEntropy logits labels
            loss.backward ()
            opt.step ()

            if epoch % 20 = 0 || epoch = 1 then
                let predicted = logits.argmax 1
                let eqSum = predicted.eq(labels).sumAll ()
                let correct = eqSum.item () |> int
                let acc = float correct / float nSamples * 100.0
                printfn "Epoch %3d  loss=%.4f  acc=%.0f%% (%d/%d)" epoch (loss.item ()) acc correct nSamples
        }

    // Test on unseen words
    printfn ""
    printfn "--- Predictions on unseen words ---"

    let testWords = [| "lovely"; "sweet"; "cruel"; "dire"; "glad"; "bleak" |]

    let testInput =
        Toro.noGrad (fun () ->
            let testData = testWords |> Array.collect (fun w -> encodeWord w)
            let t = Tensor.ofArray (testData, Cpu)
            let t = t.reshape [ testWords.Length; maxLen ]
            let logits = forward model t
            logits.argmax 1)

    for i in 0 .. testWords.Length - 1 do
        let v = testInput[i].toFloat32Scalar ()
        let label = if int v = 0 then "positive" else "negative"
        printfn "  %-10s -> %s" testWords[i] label

    0
