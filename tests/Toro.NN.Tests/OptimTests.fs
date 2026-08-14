module OptimTests

open System
open System.IO
open Xunit
open FsUnit.Xunit
open Toro
open TorchSharp
open Toro.NN
open TestHelper

let private withTempDir action =
    let dir = Path.Combine(Path.GetTempPath(), $"toro-optim-{Guid.NewGuid()}")

    try
        Directory.CreateDirectory dir |> ignore
        action dir
    finally
        if Directory.Exists dir then
            Directory.Delete(dir, true)

let private trainOnce (linear: Linear) (optimizer: #IOptimizer) x target =
    optimizer.zeroGrad ()

    let loss =
        linear.forward x
        |> fun prediction -> Loss.mse prediction target

    loss.backward ()
    optimizer.step ()

[<Fact>]
let ``SGD step reduces loss`` () =
    let linear = Linear.init 4 2 torch.float32 torch.CPU

    let x = torch.randn ([| 8L; 4L |], dtype = torch.float32, device = torch.CPU)
    let target = torch.randn ([| 8L; 2L |], dtype = torch.float32, device = torch.CPU)

    let opt = SGD.create 0.01 (Model.trainableParams linear)

    let getLoss () =
        let y = linear.forward x
        Loss.mse y target

    let loss0 = (getLoss ()).ToSingle()

    for _ in 1..20 do
        let loss = getLoss ()
        opt.zeroGrad ()
        loss.backward ()
        opt.step ()

    let lossN = (getLoss ()).ToSingle()
    lossN |> should be (lessThan loss0)

[<Fact>]
let ``optimizer creation rejects invalid named tensors`` () =
    let trainable = Init.toParam [| 2L |] torch.float32 torch.CPU (Init.Const 1.0)
    let frozen = Init.toTensor [| 2L |] torch.float32 torch.CPU (Init.Const 1.0)

    let named name kind tensor = {
        Name = name
        Tensor = tensor
        Kind = kind
    }

    Assert.Throws<ArgumentException>(fun () -> SGD.create 0.1 [ named "buffer" Buffer trainable ] |> ignore)
    |> ignore

    Assert.Throws<ArgumentException>(fun () -> SGD.create 0.1 [ named "frozen" Parameter frozen ] |> ignore)
    |> ignore

    Assert.Throws<ArgumentException>(fun () ->
        SGD.create 0.1 [
            named "same" Parameter trainable
            named "same" Parameter (torch.ones_like trainable)
        ]
        |> ignore)
    |> ignore

    Assert.Throws<ArgumentException>(fun () ->
        SGD.create 0.1 [ named "first" Parameter trainable; named "second" Parameter trainable ]
        |> ignore)
    |> ignore

[<Fact>]
let ``AdamW state uses canonical names and restores independently of parameter order`` () =
    withTempDir (fun dir ->
        let statePath = Path.Combine(dir, "optimizer.safetensors")
        let source = Linear.init 4 2 torch.float32 torch.CPU
        let sourceOptimizer = AdamW.createWithLr 0.01 (Model.trainableParams source)
        let x = torch.randn ([| 8L; 4L |], dtype = torch.float32)
        let target = torch.randn ([| 8L; 2L |], dtype = torch.float32)

        trainOnce source sourceOptimizer x target
        sourceOptimizer.saveState statePath

        let state = SafeTensors.load statePath

        state
        |> Map.keys
        |> Set.ofSeq
        |> should equal (Set [ "step"; "m.Weight"; "v.Weight"; "m.Bias"; "v.Bias" ])

        let replica = Linear.init 4 2 torch.float32 torch.CPU

        Model.namedState source
        |> List.map (fun item -> item.Name, item.Tensor)
        |> Map.ofList
        |> fun tensors -> Model.loadFromDict replica tensors None Strict
        |> ignore

        let replicaOptimizer =
            Model.trainableParams replica
            |> List.rev
            |> AdamW.createWithLr 0.01

        replicaOptimizer.loadStateDict state
        trainOnce source sourceOptimizer x target
        trainOnce replica replicaOptimizer x target

        (source.Weight - replica.Weight).abs().max().ToSingle()
        |> should (equalWithin 1e-6f) 0.0f

        (source.Bias.Value - replica.Bias.Value).abs().max().ToSingle()
        |> should (equalWithin 1e-6f) 0.0f)

[<Fact>]
let ``AdamW state validation rejects missing unexpected shape and dtype mismatches`` () =
    withTempDir (fun dir ->
        let statePath = Path.Combine(dir, "optimizer.safetensors")
        let linear = Linear.init 4 2 torch.float32 torch.CPU
        let optimizer = AdamW.createWithLr 0.01 (Model.trainableParams linear)
        optimizer.saveState statePath
        let state = SafeTensors.load statePath

        Assert.Throws<InvalidOperationException>(fun () -> optimizer.validateStateDict (Map.remove "m.Weight" state))
        |> ignore

        Assert.Throws<InvalidOperationException>(fun () ->
            optimizer.validateStateDict (Map.add "unexpected" (torch.ones ([| 1L |])) state))
        |> ignore

        Assert.Throws<InvalidOperationException>(fun () ->
            optimizer.validateStateDict (Map.add "m.Weight" (torch.ones ([| 1L |])) state))
        |> ignore

        let wrongDType = torch.zeros (state["m.Weight"].shape, dtype = torch.float64)

        Assert.Throws<InvalidOperationException>(fun () -> optimizer.validateStateDict (Map.add "m.Weight" wrongDType state))
        |> ignore)

[<Fact>]
let ``AdamW step reduces loss`` () =
    let linear = Linear.init 4 2 torch.float32 torch.CPU

    let x = torch.randn ([| 8L; 4L |], dtype = torch.float32, device = torch.CPU)
    let target = torch.randn ([| 8L; 2L |], dtype = torch.float32, device = torch.CPU)

    let opt = AdamW.createWithLr 0.01 (Model.trainableParams linear)


    let getLoss () =
        let y = linear.forward x
        Loss.mse y target

    let loss0 = (getLoss ()).ToSingle()

    for _ in 1..20 do
        let loss = getLoss ()
        opt.zeroGrad ()
        loss.backward ()
        opt.step ()

    let lossN = (getLoss ()).ToSingle()
    lossN |> should be (lessThan loss0)
