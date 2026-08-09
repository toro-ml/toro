module LossTests

open Xunit
open FsUnit.Xunit
open Toro
open Toro.NN
open TestHelper

[<Fact>]
let ``MSE loss returns scalar`` () =
    let inp = Tensor.randn ([ 4; 3 ], F32, Cpu) |> unwrap
    let target = Tensor.randn ([ 4; 3 ], F32, Cpu) |> unwrap

    let loss = Loss.mse inp target |> unwrap
    loss.Shape |> should equal List.empty<int>

    let v = loss.toFloat32Scalar () |> unwrap
    v |> should be (greaterThan 0.0f)

[<Fact>]
let ``MSE loss of identical tensors is zero`` () =
    let t = Tensor.ones ([ 2; 3 ], F32, Cpu) |> unwrap

    let loss = Loss.mse t t |> unwrap
    let v = loss.toFloat32Scalar () |> unwrap
    v |> should be (lessThan 1e-6f)

[<Fact>]
let ``Cross entropy loss returns scalar`` () =
    let logits = Tensor.randn ([ 4; 3 ], F32, Cpu) |> unwrap
    let targets = Tensor.zeros ([ 4 ], I64, Cpu) |> unwrap

    let loss = Loss.crossEntropy logits targets |> unwrap
    loss.Shape |> should equal List.empty<int>

    let v = loss.toFloat32Scalar () |> unwrap
    v |> should be (greaterThan 0.0f)

[<Fact>]
let ``NLL loss returns scalar`` () =
    let logProbs = Tensor.randn ([ 4; 3 ], F32, Cpu) |> unwrap
    let logSm = logProbs.logSoftmax -1 |> unwrap
    let targets = Tensor.zeros ([ 4; 1 ], I64, Cpu) |> unwrap

    let loss = Loss.nll logSm targets |> unwrap
    loss.Shape |> should equal List.empty<int>

[<Fact>]
let ``Binary cross-entropy with logit returns scalar`` () =
    let inp = Tensor.randn ([ 8; 1 ], F32, Cpu) |> unwrap
    let target = Tensor.zeros ([ 8; 1 ], F32, Cpu) |> unwrap

    let loss = Loss.binaryCrossEntropyWithLogit inp target |> unwrap
    loss.Shape |> should equal List.empty<int>
    let v = loss.toFloat32Scalar () |> unwrap
    v |> should be (greaterThan 0.0f)

[<Fact>]
let ``L1 loss returns scalar`` () =
    let inp = Tensor.randn ([ 4; 3 ], F32, Cpu) |> unwrap
    let target = Tensor.randn ([ 4; 3 ], F32, Cpu) |> unwrap

    let loss = Loss.l1 inp target |> unwrap
    loss.Shape |> should equal List.empty<int>
    let v = loss.toFloat32Scalar () |> unwrap
    v |> should be (greaterThan 0.0f)

[<Fact>]
let ``L1 loss of identical tensors is zero`` () =
    let t = Tensor.ones ([ 2; 3 ], F32, Cpu) |> unwrap

    let loss = Loss.l1 t t |> unwrap
    let v = loss.toFloat32Scalar () |> unwrap
    v |> should be (lessThan 1e-6f)

[<Fact>]
let ``Smooth L1 loss returns scalar`` () =
    let inp = Tensor.randn ([ 4; 3 ], F32, Cpu) |> unwrap
    let target = Tensor.randn ([ 4; 3 ], F32, Cpu) |> unwrap

    let loss = Loss.smoothL1 1.0 inp target |> unwrap
    loss.Shape |> should equal List.empty<int>
    let v = loss.toFloat32Scalar () |> unwrap
    v |> should be (greaterThan 0.0f)

[<Fact>]
let ``Smooth L1 loss of identical tensors is zero`` () =
    let t = Tensor.ones ([ 2; 3 ], F32, Cpu) |> unwrap

    let loss = Loss.smoothL1 1.0 t t |> unwrap
    let v = loss.toFloat32Scalar () |> unwrap
    v |> should be (lessThan 1e-6f)

[<Fact>]
let ``KL divergence loss returns scalar`` () =
    let logProbs = Tensor.randn ([ 4; 3 ], F32, Cpu) |> unwrap
    let inp = logProbs.logSoftmax -1 |> unwrap
    let target = logProbs.softmax -1 |> unwrap

    let loss = Loss.klDiv inp target |> unwrap
    loss.Shape |> should equal List.empty<int>
