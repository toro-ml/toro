open System
open System.IO
open TorchSharp
open Toro
open Toro.NN

do TorchSharp.torchvision.io.DefaultImager <- TorchSharp.torchvision.io.SkiaImager()

let latentDim = 64

type Gan = { Gen: Sequential; Disc: Sequential }

let createGan () =
    result {
        let! gL1 = Linear.init latentDim 256 F32 Cpu
        let! gL2 = Linear.init 256 512 F32 Cpu
        let! gL3 = Linear.init 512 784 F32 Cpu

        let! dL1 = Linear.init 784 512 F32 Cpu
        let! dL2 = Linear.init 512 256 F32 Cpu
        let! dL3 = Linear.init 256 1 F32 Cpu

        return {
            Gen =
                sequential {
                    gL1
                    LeakyRelu 0.2
                    gL2
                    LeakyRelu 0.2
                    gL3
                    Tanh
                }
            Disc =
                sequential {
                    dL1
                    LeakyRelu 0.2
                    dL2
                    LeakyRelu 0.2
                    dL3
                }
        }
    }

let generate (gen: Sequential) (batchSize: int) =
    result {
        let! z = Tensor.randn ([ batchSize; latentDim ], F32, Cpu)
        return! gen.forward z
    }

[<EntryPoint>]
let main _argv =
    result {
        let batchSize = 128
        let epochs = 30
        let lrG = 2e-4
        let lrD = 2e-4

        let dataPath =
            IO.Path.Combine(Environment.GetFolderPath Environment.SpecialFolder.LocalApplicationData, "toro-mnist")

        printfn "Loading MNIST dataset..."

        use trainDataset: torch.utils.data.Dataset =
            TorchSharp.torchvision.datasets.MNIST(dataPath, true, download = true)

        printfn "  Train samples: %d" trainDataset.Count

        let! gan = createGan ()
        let! optG = AdamW.createWithLr lrG (Model.trainableVars gan.Gen)
        let! optD = AdamW.createWithLr lrD (Model.trainableVars gan.Disc)

        printfn ""
        printfn "Generator:     z(%d) -> 256 -> 512 -> 784 (tanh)" latentDim
        printfn "Discriminator: 784 -> 512 -> 256 -> 1 (logit)"
        printfn "Optimizer: AdamW (lr_G=%.0e, lr_D=%.0e)" lrG lrD
        printfn ""

        use trainLoader =
            torch.utils.data.DataLoader(trainDataset, batchSize, shuffle = true, device = torch.CPU)

        for epoch in 1..epochs do
            let mutable dLossSum = 0.0
            let mutable gLossSum = 0.0
            let mutable steps = 0

            for batch in trainLoader do
                let images = batch["data"]
                let n = int images.shape[0]

                // --- Train Discriminator ---
                let! real = Tensor.ofTorchTensor images
                let! real = real.flatten (1, -1)
                // Scale [0,1] to [-1,1] to match generator's tanh output
                let! real = real.affine (2.0, -1.0)

                let! realLogits = gan.Disc.forward real
                let! onesTarget = Tensor.ones ([ n; 1 ], F32, Cpu)
                let! dLossReal = Loss.binaryCrossEntropyWithLogit realLogits onesTarget

                let! fake = generate gan.Gen n
                let! fake = fake.detach ()
                let! fakeLogits = gan.Disc.forward fake
                let! zerosTarget = Tensor.zeros ([ n; 1 ], F32, Cpu)
                let! dLossFake = Loss.binaryCrossEntropyWithLogit fakeLogits zerosTarget

                let! dLoss = (dLossReal + dLossFake) *~. 0.5
                optD.zeroGrad ()
                do! dLoss.backward ()
                do! optD.step ()

                // --- Train Generator ---
                let! fake = generate gan.Gen n
                let! fakeLogits = gan.Disc.forward fake
                let! gLoss = Loss.binaryCrossEntropyWithLogit fakeLogits onesTarget

                optG.zeroGrad ()
                do! gLoss.backward ()
                do! optG.step ()

                dLossSum <- dLossSum + dLoss.item ()
                gLossSum <- gLossSum + gLoss.item ()
                steps <- steps + 1

            let dAvg = dLossSum / float steps
            let gAvg = gLossSum / float steps
            printfn "Epoch %2d/%d  D_loss=%.4f  G_loss=%.4f" epoch epochs dAvg gAvg

        // Save generated samples
        printfn ""

        let! samples =
            Toro.noGrad (fun () ->
                result {
                    let! fake = generate gan.Gen 64
                    // Scale [-1,1] back to [0,1]
                    return! fake.affine (0.5, 0.5)
                })

        let! grid = samples.reshape [ 64; 1; 28; 28 ]
        let outPath = Path.Combine(__SOURCE_DIRECTORY__, "generated.png")

        TorchSharp.torchvision.utils.save_image (grid.Inner, outPath, TorchSharp.torchvision.ImageFormat.Png, nrow = 8L)

        printfn "Saved %s" outPath
        printfn "Done."
    }
    |> function
        | Ok() -> 0
        | Error e ->
            eprintfn "%A" e
            1
