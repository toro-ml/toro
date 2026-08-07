open System
open TorchSharp
open Toro
open Toro.NN

type CnnModel = {
    Features: SequentialT
    Classifier: SequentialT
} with

    member this.forwardT (x: Tensor) (train: bool) : Result<Tensor, ToroError> =
        result {
            let! x = this.Features.forwardT x train
            return! this.Classifier.forwardT x train
        }

    interface IModuleT with
        member this.forwardT x train = this.forwardT x train

let createModel () =
    let pad1 = {
        Conv2dConfig.defaultConfig with
            Padding = 1
    }

    result {
        let! conv1 = Conv2d.init 1 32 3 pad1 F32 Cpu
        let! bn1 = BatchNorm.initDefault 32 F32 Cpu
        let! conv2 = Conv2d.init 32 64 3 pad1 F32 Cpu
        let! bn2 = BatchNorm.initDefault 64 F32 Cpu
        let pool = MaxPool2d.createDefault 2
        let! fc1 = Linear.init (64 * 7 * 7) 128 F32 Cpu
        let drop = Dropout.create 0.5
        let! fc2 = Linear.init 128 10 F32 Cpu

        return {
            Features =
                sequentialT {
                    conv1
                    bn1
                    Relu
                    pool
                    conv2
                    bn2
                    Relu
                    pool
                }
            Classifier =
                sequentialT {
                    Func.create (fun x -> x.flatten (1, -1))
                    fc1
                    Relu
                    drop
                    fc2
                }
        }
    }

[<EntryPoint>]
let main _argv =
    result {
        let batchSize = 128
        let epochs = 5
        let lr = 1e-3

        let dataPath =
            IO.Path.Combine(Environment.GetFolderPath Environment.SpecialFolder.LocalApplicationData, "toro-mnist")

        printfn "Loading MNIST dataset..."

        let norm = TorchSharp.torchvision.transforms.Normalize([| 0.1307 |], [| 0.3081 |])

        use trainDataset: torch.utils.data.Dataset =
            TorchSharp.torchvision.datasets.MNIST(dataPath, true, download = true, target_transform = norm)

        use testDataset: torch.utils.data.Dataset =
            TorchSharp.torchvision.datasets.MNIST(dataPath, false, download = true, target_transform = norm)

        printfn "  Train samples: %d" trainDataset.Count
        printfn "  Test samples:  %d" testDataset.Count

        let! model = createModel ()
        let! opt = AdamW.createWithLr lr (Model.trainableVars model)
        let opt = opt :> IOptimizer

        printfn ""
        printfn "Model: Conv2d(1->32) -> BN -> Conv2d(32->64) -> BN -> FC(3136->128) -> Dropout -> FC(128->10)"
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

                let! x = Tensor.ofTorchTensor images
                let! target = Tensor.ofTorchTensor labels
                let! logits = model.forwardT x true
                let! loss = Loss.crossEntropy logits target
                do! opt.backwardStep loss

                let lossVal = loss.item ()
                let! predicted = logits.argmax 1
                let! eqSum = predicted.eq(target).sumAll ()
                let correct = eqSum.item () |> int64
                let n = images.shape[0]

                totalLoss <- totalLoss + float lossVal * float n
                totalCorrect <- totalCorrect + correct
                totalSamples <- totalSamples + n

            let avgLoss = totalLoss / float totalSamples
            let accuracy = float totalCorrect / float totalSamples * 100.0
            printf "Epoch %d/%d  train loss=%.4f  acc=%.1f%%" epoch epochs avgLoss accuracy

            let mutable testCorrect = 0L
            let mutable testTotal = 0L

            do!
                Toro.noGrad (fun () ->
                    result {
                        for batch in testLoader do
                            let images = batch["data"]
                            let labels = batch["label"]

                            let! x = Tensor.ofTorchTensor images
                            let! target = Tensor.ofTorchTensor labels
                            let! logits = model.forwardT x false
                            let! predicted = logits.argmax 1
                            let! eqSum = predicted.eq(target).sumAll ()
                            let correct = eqSum.item () |> int64
                            let n = images.shape[0]

                            testCorrect <- testCorrect + correct
                            testTotal <- testTotal + n
                    })

            let testAcc = float testCorrect / float testTotal * 100.0
            printfn "  test acc=%.1f%%" testAcc

        printfn ""
        printfn "Done."
    }

    |> function
        | Ok() -> 0
        | Error e ->
            eprintfn "%A" e
            1
