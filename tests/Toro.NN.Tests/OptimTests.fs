module OptimTests

open Xunit
open FsUnit.Xunit
open Toro
open Toro.NN
open TestHelper

[<Fact>]
let ``SGD step reduces loss`` () =
    let linear = Linear.init 4 2 F32 Cpu

    let x = Tensor.randn ([ 8; 4 ], F32, Cpu)
    let target = Tensor.randn ([ 8; 2 ], F32, Cpu)

    let opt = SGD.create 0.01 (Model.trainableVars linear)

    let getLoss () =
        let y = linear.forward x
        Loss.mse y target

    let loss0 = (getLoss ()).toFloat32Scalar ()

    for _ in 1..20 do
        let loss = getLoss ()
        opt.zeroGrad ()
        loss.backward ()
        opt.step ()

    let lossN = (getLoss ()).toFloat32Scalar ()
    lossN |> should be (lessThan loss0)

[<Fact>]
let ``AdamW step reduces loss`` () =
    let linear = Linear.init 4 2 F32 Cpu

    let x = Tensor.randn ([ 8; 4 ], F32, Cpu)
    let target = Tensor.randn ([ 8; 2 ], F32, Cpu)

    let opt = AdamW.createWithLr 0.01 (Model.trainableVars linear)


    let getLoss () =
        let y = linear.forward x
        Loss.mse y target

    let loss0 = (getLoss ()).toFloat32Scalar ()

    for _ in 1..20 do
        let loss = getLoss ()
        opt.zeroGrad ()
        loss.backward ()
        opt.step ()

    let lossN = (getLoss ()).toFloat32Scalar ()
    lossN |> should be (lessThan loss0)
