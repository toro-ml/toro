module OptimTests

open Xunit
open FsUnit.Xunit
open Toro
open Toro.NN
open TestHelper

[<Fact>]
let ``SGD step reduces loss`` () =
    let linear = Linear.init 4 2 F32 Cpu |> unwrap

    let x = Tensor.randn ([ 8; 4 ], F32, Cpu) |> unwrap
    let target = Tensor.randn ([ 8; 2 ], F32, Cpu) |> unwrap

    let opt = SGD.create 0.01 (Model.trainableVars linear) :> IOptimizer

    let getLoss () =
        result {
            let! y = linear.forward x
            return! Loss.mse y target
        }
        |> unwrap

    let loss0 = (getLoss ()).toFloat32Scalar () |> unwrap

    for _ in 1..20 do
        let loss = getLoss ()
        opt.backwardStep loss |> unwrap

    let lossN = (getLoss ()).toFloat32Scalar () |> unwrap
    lossN |> should be (lessThan loss0)

[<Fact>]
let ``AdamW step reduces loss`` () =
    let linear = Linear.init 4 2 F32 Cpu |> unwrap

    let x = Tensor.randn ([ 8; 4 ], F32, Cpu) |> unwrap
    let target = Tensor.randn ([ 8; 2 ], F32, Cpu) |> unwrap

    let opt =
        AdamW.createWithLr 0.01 (Model.trainableVars linear)
        |> unwrap
        :> IOptimizer

    let getLoss () =
        result {
            let! y = linear.forward x
            return! Loss.mse y target
        }
        |> unwrap

    let loss0 = (getLoss ()).toFloat32Scalar () |> unwrap

    for _ in 1..20 do
        let loss = getLoss ()
        opt.backwardStep loss |> unwrap

    let lossN = (getLoss ()).toFloat32Scalar () |> unwrap
    lossN |> should be (lessThan loss0)
