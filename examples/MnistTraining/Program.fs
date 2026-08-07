open System
open TorchSharp
open Toro
open Toro.NN

let unwrap r =
    match r with
    | Ok v -> v
    | Error e -> failwithf "%A" e

type MnistModel = {
    Conv1: Conv2d
    Conv2: Conv2d
    Fc1: Linear
    Fc2: Linear
} with

    member this.forward(x: Tensor) : Result<Tensor, ToroError> =
        result {
            let! x = this.Conv1.forward x
            let! x = x.relu ()
            let! x = this.Conv2.forward x
            let! x = x.relu ()
            let! x = x.flatten (1, -1)
            let! x = this.Fc1.forward x
            let! x = x.relu ()
            return! this.Fc2.forward x
        }

    interface IModule with
        member this.forward x = this.forward x

let createModel () =
    let stride2 = {
        Conv2dConfig.defaultConfig with
            Stride = 2
    }

    result {
        let! conv1 = Conv2d.init 1 8 5 stride2 F32 Cpu
        let! conv2 = Conv2d.init 8 16 5 stride2 F32 Cpu
        let! fc1 = Linear.init 256 64 F32 Cpu
        let! fc2 = Linear.init 64 10 F32 Cpu

        return {
            Conv1 = conv1
            Conv2 = conv2
            Fc1 = fc1
            Fc2 = fc2
        }
    }

[<EntryPoint>]
let main _argv =
    let batchSize = 64
    let epochs = 5
    let lr = 1e-3

    let dataPath =
        IO.Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "toro-mnist")

    printfn "Loading MNIST dataset..."

    let norm = TorchSharp.torchvision.transforms.Normalize([| 0.1307 |], [| 0.3081 |])

    use trainDataset: torch.utils.data.Dataset =
        TorchSharp.torchvision.datasets.MNIST(dataPath, true, download = true, target_transform = norm)

    use testDataset: torch.utils.data.Dataset =
        TorchSharp.torchvision.datasets.MNIST(dataPath, false, download = true, target_transform = norm)

    printfn "  Train samples: %d" trainDataset.Count
    printfn "  Test samples:  %d" testDataset.Count

    let model = createModel () |> unwrap

    let opt = AdamW.createWithLr lr (Model.trainableVars model) |> unwrap :> IOptimizer

    printfn ""

    printfn "Model: Conv2d(1->8, 5, s2) -> Conv2d(8->16, 5, s2) -> Linear(256->64) -> Linear(64->10)"

    printfn "Optimizer: AdamW (lr=%.0e)" lr
    printfn ""

    use trainLoader =
        torch.utils.data.DataLoader(trainDataset, batchSize, shuffle = true, device = torch.CPU)

    use testLoader = torch.utils.data.DataLoader(testDataset, 256, device = torch.CPU)

    for epoch in 1..epochs do
        let mutable totalLoss = 0.0
        let mutable totalCorrect = 0L
        let mutable totalSamples = 0L

        for batch in trainLoader do
            let images = batch["data"]
            let labels = batch["label"]

            let trainResult =
                result {
                    let! x = Tensor.ofTorchTensor images
                    let! target = Tensor.ofTorchTensor labels
                    let! logits = model.forward x
                    let! loss = Loss.crossEntropy logits target

                    opt.backwardStep loss |> unwrap

                    let lossVal = loss.toFloat32Scalar () |> unwrap
                    let predicted = logits.Inner.argmax (1L)
                    let correct = predicted.eq(labels).sum().ToInt64()

                    return lossVal, correct, images.shape[0]
                }

            let lossVal, correct, n = unwrap trainResult
            totalLoss <- totalLoss + float lossVal * float n
            totalCorrect <- totalCorrect + correct
            totalSamples <- totalSamples + n

        let avgLoss = totalLoss / float totalSamples
        let accuracy = float totalCorrect / float totalSamples * 100.0
        printf "Epoch %d/%d  train loss=%.4f  acc=%.1f%%" epoch epochs avgLoss accuracy

        let mutable testCorrect = 0L
        let mutable testTotal = 0L

        Toro.noGrad (fun () ->
            for batch in testLoader do
                let images = batch["data"]
                let labels = batch["label"]

                let logits =
                    result {
                        let! x = Tensor.ofTorchTensor images
                        return! model.forward x
                    }
                    |> unwrap

                let predicted = logits.Inner.argmax (1L)
                testCorrect <- testCorrect + predicted.eq(labels).sum().ToInt64()
                testTotal <- testTotal + images.shape[0])

        let testAcc = float testCorrect / float testTotal * 100.0
        printfn "  test acc=%.1f%%" testAcc

    printfn ""
    printfn "Done."
    0
