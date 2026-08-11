module OptimTests

open Xunit
open FsUnit.Xunit
open Toro
open TorchSharp
open Toro.NN
open TestHelper

[<Fact>]
let ``SGD step reduces loss`` () =
    let linear = Linear.init 4 2 torch.float32 torch.CPU

    let x = torch.randn ([| 8L; 4L |], dtype = torch.float32, device = torch.CPU)
    let target = torch.randn ([| 8L; 2L |], dtype = torch.float32, device = torch.CPU)

    let opt = SGD.create 0.01 (Model.trainableVars linear)

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
let ``AdamW step reduces loss`` () =
    let linear = Linear.init 4 2 torch.float32 torch.CPU

    let x = torch.randn ([| 8L; 4L |], dtype = torch.float32, device = torch.CPU)
    let target = torch.randn ([| 8L; 2L |], dtype = torch.float32, device = torch.CPU)

    let opt = AdamW.createWithLr 0.01 (Model.trainableVars linear)


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
