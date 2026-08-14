open System
open TorchSharp
open Toro
open Toro.NN
open Toro.Vision

type Features = {
    Conv1: Conv2d
    Bn1: BatchNorm
    Pool1: MaxPool2d
    Conv2: Conv2d
    Bn2: BatchNorm
    Pool2: MaxPool2d
} with

    member this.forward(train: bool) : Tensor -> Tensor =
        this.Conv1.forward
        >> this.Bn1.forwardT train
        >> _.relu()
        >> this.Pool1.forward
        >> this.Conv2.forward
        >> this.Bn2.forwardT train
        >> _.relu()
        >> this.Pool2.forward

type Classifier = {
    Fc1: Linear
    Drop: Dropout
    Fc2: Linear
} with

    member this.forward(train: bool) : Tensor -> Tensor =
        _.flatten(1L, -1L)
        >> this.Fc1.forward
        >> _.relu()
        >> this.Drop.forwardT train
        >> this.Fc2.forward

type CnnModel = {
    Features: Features
    Classifier: Classifier
} with

    member this.forward(train: bool) : Tensor -> Tensor =
        this.Features.forward train >> this.Classifier.forward train

let createModel () =
    let pad1 = {
        Conv2dConfig.defaultConfig with
            Padding = 1
    }

    let conv1 = Conv2d.init 1 32 3 pad1 torch.float32 torch.CPU
    let bn1 = BatchNorm.initDefault 32 torch.float32 torch.CPU
    let conv2 = Conv2d.init 32 64 3 pad1 torch.float32 torch.CPU
    let bn2 = BatchNorm.initDefault 64 torch.float32 torch.CPU
    let pool = MaxPool2d.createDefault 2
    let fc1 = Linear.init (64L * 7L * 7L) 128L torch.float32 torch.CPU
    let drop = Dropout.create 0.5
    let fc2 = Linear.init 128 10 torch.float32 torch.CPU

    {
        Features = {
            Conv1 = conv1
            Bn1 = bn1
            Pool1 = pool
            Conv2 = conv2
            Bn2 = bn2
            Pool2 = pool
        }
        Classifier = { Fc1 = fc1; Drop = drop; Fc2 = fc2 }
    }

[<EntryPoint>]
let main _argv =
    let batchSize = 128
    let epochs = 5
    let lr = 1e-3

    let dataPath =
        IO.Path.Combine(Environment.GetFolderPath Environment.SpecialFolder.LocalApplicationData, "toro-mnist")

    printfn "Loading MNIST dataset..."

    let mnistNorm: Normalize = { Mean = [ 0.1307 ]; Std = [ 0.3081 ] }

    use trainDataset: torch.utils.data.Dataset =
        TorchSharp.torchvision.datasets.MNIST(dataPath, true, download = true)

    use testDataset: torch.utils.data.Dataset =
        TorchSharp.torchvision.datasets.MNIST(dataPath, false, download = true)

    printfn "  Train samples: %d" trainDataset.Count
    printfn "  Test samples:  %d" testDataset.Count

    let model = createModel ()
    let opt = AdamW.createWithLr lr (Model.trainableParams model)

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
            scoped {
                let images = batch["data"]
                let labels = batch["label"]

                let x = mnistNorm.apply images
                let target = labels
                opt.zeroGrad ()
                let logits = model.forward true x
                let loss = Loss.crossEntropy logits target
                loss.backward ()
                opt.step ()

                let lossVal = loss.item<float> ()
                let predicted = logits.argmax 1L
                let eqSum = predicted.eq(target).sum ()
                let correct = eqSum.item<int64> ()
                let n = images.shape[0]

                totalLoss <- totalLoss + float lossVal * float n
                totalCorrect <- totalCorrect + correct
                totalSamples <- totalSamples + n
            }

        let avgLoss = totalLoss / float totalSamples
        let accuracy = float totalCorrect / float totalSamples * 100.0
        printf "Epoch %d/%d  train loss=%.4f  acc=%.1f%%" epoch epochs avgLoss accuracy

        let mutable testCorrect = 0L
        let mutable testTotal = 0L

        Toro.noGrad (fun () ->
            for batch in testLoader do
                scoped {
                    let images = batch["data"]
                    let labels = batch["label"]

                    let x = mnistNorm.apply images
                    let target = labels
                    let logits = model.forward false x
                    let predicted = logits.argmax 1L
                    let eqSum = predicted.eq(target).sum ()
                    let correct = eqSum.item<int64> ()
                    let n = images.shape[0]

                    testCorrect <- testCorrect + correct
                    testTotal <- testTotal + n
                })

        let testAcc = float testCorrect / float testTotal * 100.0
        printfn "  test acc=%.1f%%" testAcc

    printfn ""
    printfn "Done."
    0
