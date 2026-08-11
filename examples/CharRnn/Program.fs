open TorchSharp
open Toro
open Toro.NN

let corpus =
    "to be or not to be that is the question "
    + "whether tis nobler in the mind to suffer "
    + "the slings and arrows of outrageous fortune "
    + "or to take arms against a sea of troubles "
    + "and by opposing end them to die to sleep "
    + "no more and by a sleep to say we end "
    + "the heartache and the thousand natural shocks "
    + "that flesh is heir to tis a consummation "

let chars = corpus |> Seq.distinct |> Seq.sort |> Seq.toArray
let vocabSize = chars.Length
let charToIdx = chars |> Array.mapi (fun i c -> c, i) |> Map.ofArray
let idxToChar c = chars[c]

let encode (s: string) =
    s |> Seq.map (fun c -> int64 charToIdx[c]) |> Seq.toArray

type CharRnnModel = {
    Embed: Embedding
    Lstm: LSTM
    Output: Linear
}

let embedDim = 32
let hiddenDim = 128

let createModel () =
    let embed = Embedding.init vocabSize embedDim torch.float32 torch.CPU
    let lstm = LSTM.initDefault embedDim hiddenDim torch.float32 torch.CPU
    let output = Linear.init hiddenDim vocabSize torch.float32 torch.CPU

    {
        Embed = embed
        Lstm = lstm
        Output = output
    }

let forward (model: CharRnnModel) (input: Tensor) =
    let embedded = model.Embed.forward input
    let states = RNN.scan model.Lstm.zeroState model.Lstm.step embedded
    let hidden = torch.stack (states |> List.map _.H |> List.toArray, int64 1)
    let batchSeq = int hidden.shape[0] * int hidden.shape[1]
    let flat = hidden.reshape ([| int64 batchSeq; int64 hiddenDim |])
    model.Output.forward flat

let generate (model: CharRnnModel) (seed: char) (length: int) =
    let mutable state = model.Lstm.zeroState 1

    let mutable idx = charToIdx[seed]
    let buf = System.Text.StringBuilder()
    buf.Append seed |> ignore

    for _ in 1..length do
        let inputT =
            torch.tensor ([| int64 idx |], dtype = torch.int64, device = torch.CPU)
            |> fun t -> t.unsqueeze (int64 0)

        let embedded = model.Embed.forward inputT
        let step_input = embedded.at [ I 0; I 0 ]
        let newState = model.Lstm.step step_input state
        state <- newState

        let logits = model.Output.forward newState.H
        let probs = torch.nn.functional.softmax (logits, int64 -1)
        let sampled = torch.multinomial (probs, 1L)
        idx <- int (sampled.item<int64> ())
        buf.Append(idxToChar idx) |> ignore

    buf.ToString()

[<EntryPoint>]
let main _argv =
    let seqLen = 32
    let epochs = 50
    let lr = 3e-3

    let encoded = encode corpus
    let nChunks = (encoded.Length - 1) / seqLen

    printfn "Corpus: %d chars, vocab: %d, chunks: %d" corpus.Length vocabSize nChunks

    printfn "Model: Embedding(%d,%d) -> LSTM(%d,%d) -> Linear(%d,%d)" vocabSize embedDim embedDim hiddenDim hiddenDim vocabSize

    printfn ""

    let model = createModel ()
    let opt = AdamW.createWithLr lr (Model.trainableVars model)

    for epoch in 1..epochs do
        let mutable totalLoss = 0.0

        for i in 0 .. nChunks - 1 do
            scoped {
                let offset = i * seqLen
                let inputArr = encoded[offset .. offset + seqLen - 1]
                let targetArr = encoded[offset + 1 .. offset + seqLen]

                let input =
                    torch.tensor (inputArr, dtype = torch.int64, device = torch.CPU)
                    |> fun t -> t.unsqueeze (int64 0)

                let target = torch.tensor (targetArr, dtype = torch.int64, device = torch.CPU)

                opt.zeroGrad ()
                let logits = forward model input
                let loss = Loss.crossEntropy logits target
                loss.backward ()
                opt.step ()

                totalLoss <- totalLoss + loss.ToDouble()
            }

        let avgLoss = totalLoss / float nChunks

        if epoch % 10 = 0 || epoch = 1 then
            let sample = Toro.noGrad (fun () -> generate model 't' 80)

            printfn "Epoch %2d/%d  loss=%.4f  \"%s\"" epoch epochs avgLoss sample

    printfn ""
    printfn "--- Final generation ---"

    let final = Toro.noGrad (fun () -> generate model 't' 200)
    printfn "%s" final
    0
