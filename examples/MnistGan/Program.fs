open System
open System.IO
open TorchSharp
open Toro
open Toro.NN
open Toro.Vision

let latentDim = 64

type Gan = { Gen: Sequential; Disc: Sequential }

let createGan () =
    let gL1 = Linear.init latentDim 256 torch.float32 torch.CPU
    let gL2 = Linear.init 256 512 torch.float32 torch.CPU
    let gL3 = Linear.init 512 784 torch.float32 torch.CPU

    let dL1 = Linear.init 784 512 torch.float32 torch.CPU
    let dL2 = Linear.init 512 256 torch.float32 torch.CPU
    let dL3 = Linear.init 256 1 torch.float32 torch.CPU

    {
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

let generate (gen: Sequential) (batchSize: int64) =
    let z =
        torch.randn ([| batchSize; latentDim |], dtype = torch.float32, device = torch.CPU)

    gen.forward z

[<EntryPoint>]
let main _argv =
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

    let gan = createGan ()
    let optG = AdamW.createWithLr lrG (Model.trainableParams gan.Gen)
    let optD = AdamW.createWithLr lrD (Model.trainableParams gan.Disc)

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
            scoped {
                let images = batch["data"]
                let n = images.shape[0]

                // --- Train Discriminator ---
                let real = images.flatten (1L, -1L)
                let real = real * scalar 2.0 + scalar (-1.0)

                let realLogits = gan.Disc.forward real

                let onesTarget = torch.ones ([| n; 1L |], dtype = torch.float32, device = torch.CPU)

                let dLossReal = Loss.binaryCrossEntropyWithLogit realLogits onesTarget

                let fake = generate gan.Gen n
                let fake = fake.detach ()
                let fakeLogits = gan.Disc.forward fake

                let zerosTarget =
                    torch.zeros ([| n; 1L |], dtype = torch.float32, device = torch.CPU)

                let dLossFake = Loss.binaryCrossEntropyWithLogit fakeLogits zerosTarget

                let dLoss = (dLossReal + dLossFake) * 0.5
                optD.zeroGrad ()
                dLoss.backward ()
                optD.step ()

                // --- Train Generator ---
                let fake = generate gan.Gen n
                let fakeLogits = gan.Disc.forward fake
                let gLoss = Loss.binaryCrossEntropyWithLogit fakeLogits onesTarget

                optG.zeroGrad ()
                gLoss.backward ()
                optG.step ()

                dLossSum <- dLossSum + dLoss.ToDouble()
                gLossSum <- gLossSum + gLoss.ToDouble()
                steps <- steps + 1
            }

        let dAvg = dLossSum / float steps
        let gAvg = gLossSum / float steps
        printfn "Epoch %2d/%d  D_loss=%.4f  G_loss=%.4f" epoch epochs dAvg gAvg

    // Save generated samples
    printfn ""

    let samples =
        Toro.noGrad (fun () ->
            let fake = generate gan.Gen 64L
            // Scale [-1,1] back to [0,1]
            fake * scalar 0.5 + scalar 0.5)

    let grid = samples.reshape [| 64L; 1L; 28L; 28L |]
    let outPath = Path.Combine(__SOURCE_DIRECTORY__, "generated.png")
    Image.saveGrid grid outPath Png 0 8

    printfn "Saved %s" outPath
    printfn "Done."
    0
