module LossTests

open Xunit
open FsUnit.Xunit
open Toro
open Toro.NN
open TestHelper

[<Fact>]
let ``MSE loss returns scalar`` () =
    let inp = Tensor.randn ([ 4; 3 ], F32, Cpu)
    let target = Tensor.randn ([ 4; 3 ], F32, Cpu)

    let loss = Loss.mse inp target
    loss.Shape |> should equal List.empty<int>

    let v = loss.toFloat32Scalar ()
    v |> should be (greaterThan 0.0f)

[<Fact>]
let ``MSE loss of identical tensors is zero`` () =
    let t = Tensor.ones ([ 2; 3 ], F32, Cpu)

    let loss = Loss.mse t t
    let v = loss.toFloat32Scalar ()
    v |> should be (lessThan 1e-6f)

[<Fact>]
let ``Cross entropy loss returns scalar`` () =
    let logits = Tensor.randn ([ 4; 3 ], F32, Cpu)
    let targets = Tensor.zeros ([ 4 ], I64, Cpu)

    let loss = Loss.crossEntropy logits targets
    loss.Shape |> should equal List.empty<int>

    let v = loss.toFloat32Scalar ()
    v |> should be (greaterThan 0.0f)

[<Fact>]
let ``NLL loss returns scalar`` () =
    let logProbs = Tensor.randn ([ 4; 3 ], F32, Cpu)
    let logSm = logProbs.logSoftmax -1
    let targets = Tensor.zeros ([ 4; 1 ], I64, Cpu)

    let loss = Loss.nll logSm targets
    loss.Shape |> should equal List.empty<int>

[<Fact>]
let ``Binary cross-entropy with logit returns scalar`` () =
    let inp = Tensor.randn ([ 8; 1 ], F32, Cpu)
    let target = Tensor.zeros ([ 8; 1 ], F32, Cpu)

    let loss = Loss.binaryCrossEntropyWithLogit inp target
    loss.Shape |> should equal List.empty<int>
    let v = loss.toFloat32Scalar ()
    v |> should be (greaterThan 0.0f)

[<Fact>]
let ``L1 loss returns scalar`` () =
    let inp = Tensor.randn ([ 4; 3 ], F32, Cpu)
    let target = Tensor.randn ([ 4; 3 ], F32, Cpu)

    let loss = Loss.l1 inp target
    loss.Shape |> should equal List.empty<int>
    let v = loss.toFloat32Scalar ()
    v |> should be (greaterThan 0.0f)

[<Fact>]
let ``L1 loss of identical tensors is zero`` () =
    let t = Tensor.ones ([ 2; 3 ], F32, Cpu)

    let loss = Loss.l1 t t
    let v = loss.toFloat32Scalar ()
    v |> should be (lessThan 1e-6f)

[<Fact>]
let ``Smooth L1 loss returns scalar`` () =
    let inp = Tensor.randn ([ 4; 3 ], F32, Cpu)
    let target = Tensor.randn ([ 4; 3 ], F32, Cpu)

    let loss = Loss.smoothL1 1.0 inp target
    loss.Shape |> should equal List.empty<int>
    let v = loss.toFloat32Scalar ()
    v |> should be (greaterThan 0.0f)

[<Fact>]
let ``Smooth L1 loss of identical tensors is zero`` () =
    let t = Tensor.ones ([ 2; 3 ], F32, Cpu)

    let loss = Loss.smoothL1 1.0 t t
    let v = loss.toFloat32Scalar ()
    v |> should be (lessThan 1e-6f)

[<Fact>]
let ``KL divergence loss returns scalar`` () =
    let logProbs = Tensor.randn ([ 4; 3 ], F32, Cpu)
    let inp = logProbs.logSoftmax -1
    let target = logProbs.softmax -1

    let loss = Loss.klDiv inp target
    loss.Shape |> should equal List.empty<int>
