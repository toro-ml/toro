open System
open TorchSharp
open Toro
open Toro.NN
open Toro.Vision

type Options = {
    Epochs: int
    Seed: int
    CheckpointDir: string option
    Resume: bool
}

let parseOptions (argv: string[]) =
    let rec loop options args =
        match args with
        | [] -> options
        | "--epochs" :: value :: rest -> loop { options with Epochs = int value } rest
        | "--seed" :: value :: rest -> loop { options with Seed = int value } rest
        | "--checkpoint" :: value :: rest ->
            loop
                {
                    options with
                        CheckpointDir = Some value
                }
                rest
        | "--resume" :: rest -> loop { options with Resume = true } rest
        | unknown :: _ -> invalidArg (nameof argv) $"Unknown or incomplete argument: {unknown}"

    let options =
        loop
            {
                Epochs = 5
                Seed = 1337
                CheckpointDir = None
                Resume = false
            }
            (Array.toList argv)

    if options.Epochs < 0 then
        invalidArg (nameof argv) "--epochs must be non-negative."

    if options.Resume && options.CheckpointDir.IsNone then
        invalidArg (nameof argv) "--resume requires --checkpoint <directory>."

    options

let trainingStateFileName = "training-state.safetensors"
let rngStateName = "torch.cpu_rng_state"
let schedulerStepName = "scheduler.current_step"

let saveTrainingState (scheduler: Scheduler) (checkpointDir: string) =
    let state = Scheduler.getState scheduler
    let filePath = IO.Path.Combine(checkpointDir, trainingStateFileName)

    scoped {
        let rngState = torch.get_rng_state ()

        let schedulerStep =
            torch.tensor ([| int64 state.CurrentStep |], dtype = torch.int64, device = torch.CPU)

        SafeTensors.save (Map [ rngStateName, rngState; schedulerStepName, schedulerStep ]) filePath
    }

let loadTrainingState (scheduler: Scheduler) (checkpointDir: string) =
    let filePath = IO.Path.Combine(checkpointDir, trainingStateFileName)
    let state = SafeTensors.load filePath
    let expected = Set [ rngStateName; schedulerStepName ]
    let actual = state |> Map.keys |> Set.ofSeq

    if actual <> expected then
        invalidOp $"Training state key mismatch: expected={expected}; actual={actual}."

    let rngState = state[rngStateName]
    let schedulerStep = state[schedulerStepName]

    if rngState.dtype <> torch.uint8 || rngState.ndim <> 1L then
        invalidOp "Torch CPU RNG state must be a rank-1 uint8 tensor."

    if
        schedulerStep.dtype <> torch.int64
        || schedulerStep.shape <> [| 1L |]
    then
        invalidOp "Scheduler step must be an int64 tensor with shape [1]."

    let schedulerState = {
        CurrentStep = schedulerStep.item<int64> () |> int
    }

    Scheduler.loadState schedulerState scheduler
    torch.set_rng_state rngState

    for tensor in state.Values do
        tensor.Dispose()

type MnistModel = {
    Conv1: Conv2d
    Conv2: Conv2d
    Fc1: Linear
    Fc2: Linear
} with

    member this.forward(x: Tensor) : Tensor =
        let x = this.Conv1.forward x
        let x = x.relu ()
        let x = this.Conv2.forward x
        let x = x.relu ()
        let x = x.flatten (1L, -1L)
        let x = this.Fc1.forward x
        let x = x.relu ()
        this.Fc2.forward x

    interface IModule with
        member this.forward x = this.forward x

let createModel () =
    let stride2 = {
        Conv2dConfig.defaultConfig with
            Stride = 2
    }

    let conv1 = Conv2d.init 1 8 5 stride2 torch.float32 torch.CPU
    let conv2 = Conv2d.init 8 16 5 stride2 torch.float32 torch.CPU
    let fc1 = Linear.init 256 64 torch.float32 torch.CPU
    let fc2 = Linear.init 64 10 torch.float32 torch.CPU

    {
        Conv1 = conv1
        Conv2 = conv2
        Fc1 = fc1
        Fc2 = fc2
    }

[<EntryPoint>]
let main argv =
    let options = parseOptions argv
    let batchSize = 64
    let lr = 1e-3

    torch.manual_seed (int64 options.Seed) |> ignore

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
    let scheduler = Scheduler.create (StepDecay(2, 0.5)) opt.setLearningRate lr

    let startEpoch =
        match options.Resume, options.CheckpointDir with
        | true, Some checkpointDir ->
            let completedEpoch = Checkpoint.load model opt checkpointDir
            loadTrainingState scheduler checkpointDir

            if scheduler.CurrentStep <> completedEpoch then
                invalidOp $"Checkpoint epoch {completedEpoch} does not match scheduler step {scheduler.CurrentStep}."

            completedEpoch + 1
        | _ -> 1

    printfn ""
    printfn "Model: Conv2d(1->8, 5, s2) -> Conv2d(8->16, 5, s2) -> Linear(256->64) -> Linear(64->10)"
    printfn "Optimizer: AdamW (lr=%.0e)" lr
    printfn "Seed: %d" options.Seed

    if options.Resume then
        printfn "Resumed at epoch %d" startEpoch

    printfn ""

    use testLoader = torch.utils.data.DataLoader(testDataset, 256, device = torch.CPU)

    for epoch in startEpoch .. options.Epochs do
        use trainLoader =
            torch.utils.data.DataLoader(
                trainDataset,
                batchSize,
                shuffle = true,
                device = torch.CPU,
                seed = options.Seed + epoch,
                disposeDataset = false
            )

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
                let logits = model.forward x
                let loss = Loss.crossEntropy logits target
                loss.backward ()
                opt.step ()

                let lossVal = loss.ToDouble()
                let predicted = logits.argmax 1L
                let eqSum = predicted.eq(target).sum ()
                let correct = eqSum.ToInt64()
                let n = images.shape[0]

                totalLoss <- totalLoss + float lossVal * float n
                totalCorrect <- totalCorrect + correct
                totalSamples <- totalSamples + n
            }

        let avgLoss = totalLoss / float totalSamples
        let accuracy = float totalCorrect / float totalSamples * 100.0
        printf "Epoch %d/%d  train loss=%.4f  acc=%.1f%%" epoch options.Epochs avgLoss accuracy

        let mutable testCorrect = 0L
        let mutable testTotal = 0L

        Toro.noGrad (fun () ->
            for batch in testLoader do
                scoped {
                    let images = batch["data"]
                    let labels = batch["label"]

                    let x = mnistNorm.apply images
                    let target = labels
                    let logits = model.forward x
                    let predicted = logits.argmax (int64 1)
                    let eqSum = predicted.eq(target).sum ()
                    let correct = eqSum.ToInt64()
                    let n = images.shape[0]

                    testCorrect <- testCorrect + correct
                    testTotal <- testTotal + n
                })

        let testAcc = float testCorrect / float testTotal * 100.0
        printfn "  test acc=%.1f%%" testAcc

        Scheduler.step scheduler

        match options.CheckpointDir with
        | Some checkpointDir ->
            Checkpoint.save model opt epoch checkpointDir
            saveTrainingState scheduler checkpointDir
        | None -> ()

    printfn ""
    printfn "Done."
    0
