module AutogradTests

open Xunit
open FsUnit.Xunit
open Toro
open TestHelper

[<Fact>]
let ``backward computes gradients`` () =
    let x = Tensor.ones ([ 3 ], F32, Cpu)
    let x = x.requiresGrad ()

    let y = x.mulScalar 3.0
    let loss = y.sumAll ()

    loss.backward ()

    let g = x.grad ()
    scalarF32 g |> should equal 9.0f

[<Fact>]
let ``grad returns zero-like before backward`` () =
    let x = Tensor.ones ([ 2 ], F32, Cpu)
    let x = x.requiresGrad ()

    x.zeroGrad ()
    let g = x.grad ()
    scalarF32 g |> should equal 0.0f

[<Fact>]
let ``detach removes gradient tracking`` () =
    let x = Tensor.randn ([ 2; 3 ], F32, Cpu)
    let x = x.requiresGrad ()
    let d = x.detach ()
    d.Shape |> should equal [ 2; 3 ]

[<Fact>]
let ``noGrad disables gradient tracking`` () =
    let t = Tensor.randn ([ 2; 3 ], F32, Cpu)
    let t = t.requiresGrad ()

    let y =
        Toro.noGrad (fun () ->
            let r = t.mul t
            r)

    y.Shape |> should equal [ 2; 3 ]
