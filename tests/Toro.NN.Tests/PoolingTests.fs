module PoolingTests

open Xunit
open FsUnit.Xunit
open Toro
open Toro.NN
open TestHelper

[<Fact>]
let ``MaxPool1d reduces temporal dimension`` () =
    let x = Tensor.randn ([ 1; 3; 20 ], F32, Cpu) |> unwrap
    let pool = MaxPool1d.createDefault 2
    let y = pool.forward x |> unwrap
    y.Shape |> should equal [ 1; 3; 10 ]

[<Fact>]
let ``MaxPool2d halves spatial dimensions`` () =
    let x = Tensor.randn ([ 1; 1; 8; 8 ], F32, Cpu) |> unwrap
    let pool = MaxPool2d.createDefault 2
    let y = pool.forward x |> unwrap
    y.Shape |> should equal [ 1; 1; 4; 4 ]

[<Fact>]
let ``MaxPool2d with stride and padding`` () =
    let x = Tensor.randn ([ 2; 3; 6; 6 ], F32, Cpu) |> unwrap
    let pool = MaxPool2d.create 3 1 1
    let y = pool.forward x |> unwrap
    y.Shape |> should equal [ 2; 3; 6; 6 ]

[<Fact>]
let ``AvgPool2d halves spatial dimensions`` () =
    let x = Tensor.randn ([ 1; 1; 8; 8 ], F32, Cpu) |> unwrap
    let pool = AvgPool2d.createDefault 2
    let y = pool.forward x |> unwrap
    y.Shape |> should equal [ 1; 1; 4; 4 ]

[<Fact>]
let ``MaxPool2d implements IModule`` () =
    let pool = MaxPool2d.createDefault 2
    let m = pool :> IModule
    let x = Tensor.randn ([ 1; 1; 4; 4 ], F32, Cpu) |> unwrap
    let y = m.forward x |> unwrap
    y.Shape |> should equal [ 1; 1; 2; 2 ]
