module TensorOpTests

open Xunit
open FsUnit.Xunit
open Toro
open Toro.TensorOp
open TestHelper

[<Fact>]
let ``+~ adds two results`` () =
    let a = Tensor.ones ([ 3 ], F32, Cpu)
    let b = Tensor.ones ([ 3 ], F32, Cpu)
    let c = a +~ b |> unwrap
    scalarF32 c |> should equal 6.0f

[<Fact>]
let ``-~ subtracts two results`` () =
    let a = Tensor.full ([ 3 ], 5.0, F32, Cpu)
    let b = Tensor.full ([ 3 ], 2.0, F32, Cpu)
    let c = a -~ b |> unwrap
    scalarF32 c |> should equal 9.0f

[<Fact>]
let ``*~ multiplies two results`` () =
    let a = Tensor.full ([ 3 ], 3.0, F32, Cpu)
    let b = Tensor.full ([ 3 ], 2.0, F32, Cpu)
    let c = a *~ b |> unwrap
    scalarF32 c |> should equal 18.0f

[<Fact>]
let ``/~ divides two results`` () =
    let a = Tensor.full ([ 3 ], 6.0, F32, Cpu)
    let b = Tensor.full ([ 3 ], 2.0, F32, Cpu)
    let c = a /~ b |> unwrap
    scalarF32 c |> should equal 9.0f

[<Fact>]
let ``*~. scales result by scalar`` () =
    let t = Tensor.ones ([ 3 ], F32, Cpu)
    let c = t *~. 4.0 |> unwrap
    scalarF32 c |> should equal 12.0f

[<Fact>]
let ``+~. shifts result by scalar`` () =
    let t = Tensor.ones ([ 3 ], F32, Cpu)
    let c = t +~. 2.0 |> unwrap
    scalarF32 c |> should equal 9.0f

[<Fact>]
let ``operators accept mixed Tensor and Result`` () =
    let t = Tensor.ones ([ 3 ], F32, Cpu) |> unwrap
    let r = Tensor.ones ([ 3 ], F32, Cpu)
    let c = t +~ r |> unwrap
    scalarF32 c |> should equal 6.0f
