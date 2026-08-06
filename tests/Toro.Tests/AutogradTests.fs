module AutogradTests

open Xunit
open FsUnit.Xunit
open Toro
open TestHelper

[<Fact>]
let ``backward computes gradients`` () =
    let x = Tensor.ones ([ 3 ], F32, Cpu) |> unwrap
    let x = x.requiresGrad () |> unwrap

    let y = x.mulScalar 3.0 |> unwrap
    let loss = y.sumAll () |> unwrap

    loss.backward () |> unwrap

    let g = x.grad () |> unwrap
    scalarF32 g |> should equal 9.0f

[<Fact>]
let ``grad returns zero-like before backward`` () =
    let x = Tensor.ones ([ 2 ], F32, Cpu) |> unwrap
    let x = x.requiresGrad () |> unwrap

    x.zeroGrad ()
    let g = x.grad () |> unwrap
    scalarF32 g |> should equal 0.0f

[<Fact>]
let ``detach removes gradient tracking`` () =
    let x = Tensor.randn ([ 2; 3 ], F32, Cpu) |> unwrap
    let x = x.requiresGrad () |> unwrap
    let d = x.detach () |> unwrap
    d.Shape |> should equal [ 2; 3 ]

[<Fact>]
let ``noGrad disables gradient tracking`` () =
    let t = Tensor.randn ([ 2; 3 ], F32, Cpu) |> unwrap
    let t = t.requiresGrad () |> unwrap

    let y =
        Toro.noGrad (fun () ->
            let r = t.mul t |> unwrap
            r)

    y.Shape |> should equal [ 2; 3 ]
