open System
open System.IO
open TorchSharp
open Toro
open Toro.NN

do TorchSharp.torchvision.io.DefaultImager <- TorchSharp.torchvision.io.SkiaImager()

let unwrap r =
    match r with
    | Ok v -> v
    | Error e -> failwithf "%A" e

type Autoencoder = {
    Encoder: Sequential
    Decoder: Sequential
} with

    member this.forward(x: Tensor) : Result<Tensor, ToroError> =
        result {
            let! z = this.Encoder.forward x
            return! this.Decoder.forward z
        }

    interface IModule with
        member this.forward x = this.forward x

let createModel (latentDim: int) =
    result {
        let! enc1 = Linear.init 784 256 F32 Cpu
        let! enc2 = Linear.init 256 latentDim F32 Cpu
        let! dec1 = Linear.init latentDim 256 F32 Cpu
        let! dec2 = Linear.init 256 784 F32 Cpu

        return {
            Encoder =
                sequential {
                    enc1
                    Relu
                    enc2
                }
            Decoder =
                sequential {
                    dec1
                    Relu
                    dec2
                    Sigmoid
                }
        }
    }

let preprocessBatch (images: torch.Tensor) (model: Autoencoder) =
    result {
        let! x = Tensor.ofTorchTensor images
        let! x = x.flatten (1, -1)
        let! recon = model.forward x
        return x, recon
    }

[<EntryPoint>]
let main _argv =
    let batchSize = 256
    let epochs = 10
    let lr = 1e-3
    let latentDim = 32

    let dataPath =
        IO.Path.Combine(Environment.GetFolderPath Environment.SpecialFolder.LocalApplicationData, "toro-mnist")

    printfn "Loading MNIST dataset..."

    use trainDataset: torch.utils.data.Dataset =
        TorchSharp.torchvision.datasets.MNIST(dataPath, true, download = true)

    use testDataset: torch.utils.data.Dataset =
        TorchSharp.torchvision.datasets.MNIST(dataPath, false, download = true)

    printfn "  Train samples: %d" trainDataset.Count
    printfn "  Test samples:  %d" testDataset.Count

    let model = createModel latentDim |> unwrap
    let opt = AdamW.createWithLr lr (Model.trainableVars model) |> unwrap :> IOptimizer

    printfn ""
    printfn "Autoencoder: 784 -> 256 -> %d -> 256 -> 784" latentDim
    printfn "Optimizer: AdamW (lr=%.0e)" lr
    printfn ""

    use trainLoader =
        torch.utils.data.DataLoader(trainDataset, batchSize, shuffle = true, device = torch.CPU)

    use testLoader = torch.utils.data.DataLoader(testDataset, 256, device = torch.CPU)

    for epoch in 1..epochs do
        let mutable totalLoss = 0.0
        let mutable totalSamples = 0L

        for batch in trainLoader do
            let images = batch["data"]
            let x, recon = preprocessBatch images model |> unwrap
            let loss = Loss.mse recon x |> unwrap
            opt.backwardStep loss |> unwrap

            let n = images.shape[0]
            totalLoss <- totalLoss + float (loss.item ()) * float n
            totalSamples <- totalSamples + n

        let avgLoss = totalLoss / float totalSamples
        printf "Epoch %2d/%d  train mse=%.6f" epoch epochs avgLoss

        let mutable testLoss = 0.0
        let mutable testTotal = 0L

        Toro.noGrad (fun () ->
            for batch in testLoader do
                let images = batch["data"]
                let x, recon = preprocessBatch images model |> unwrap
                let loss = Loss.mse recon x |> unwrap
                let n = images.shape[0]
                testLoss <- testLoss + float (loss.item ()) * float n
                testTotal <- testTotal + n)

        printfn "  test mse=%.6f" (testLoss / float testTotal)

    // Save reconstruction samples
    let sampleCount = 8
    let mutable saved = false

    Toro.noGrad (fun () ->
        for batch in testLoader do
            if not saved then
                let images = batch["data"]
                let orig, recon = preprocessBatch images model |> unwrap

                let toGrid (t: Tensor) =
                    t.at([ S(0, sampleCount) ]).reshape [ sampleCount; 1; 28; 28 ] |> unwrap

                let combined =
                    Tensor.cat ([ toGrid orig; toGrid recon ], 0) |> unwrap
                let outPath = Path.Combine(__SOURCE_DIRECTORY__, "reconstruction.png")

                TorchSharp.torchvision.utils.save_image (
                    combined.Inner,
                    outPath,
                    TorchSharp.torchvision.ImageFormat.Png,
                    nrow = int64 sampleCount
                )

                printfn "Saved %s" outPath
                saved <- true)

    printfn "Done."
    0
