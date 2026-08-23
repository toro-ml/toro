module LossTests

open Xunit
open FsUnit.Xunit
open Toro
open TorchSharp
open Toro.NN
open TestHelper

[<Fact>]
let ``MSE loss returns scalar`` () =
    let inp = torch.randn ([| 4L; 3L |], dtype = torch.float32, device = torch.CPU)
    let target = torch.randn ([| 4L; 3L |], dtype = torch.float32, device = torch.CPU)

    let loss = Loss.mse inp target
    loss.shape |> should equal [||]

    let v = loss.ToSingle()
    v |> should be (greaterThan 0.0f)

[<Fact>]
let ``MSE loss of identical tensors is zero`` () =
    let t = torch.ones ([| 2L; 3L |], dtype = torch.float32, device = torch.CPU)

    let loss = Loss.mse t t
    let v = loss.ToSingle()
    v |> should be (lessThan 1e-6f)

[<Fact>]
let ``Cross entropy loss returns scalar`` () =
    let logits = torch.randn ([| 4L; 3L |], dtype = torch.float32, device = torch.CPU)
    let targets = torch.zeros ([| 4L |], dtype = torch.int64, device = torch.CPU)

    let loss = Loss.crossEntropy logits targets
    loss.shape |> should equal [||]

    let v = loss.ToSingle()
    v |> should be (greaterThan 0.0f)

[<Fact>]
let ``NLL loss returns scalar`` () =
    let logProbs = torch.randn ([| 4L; 3L |], dtype = torch.float32, device = torch.CPU)
    let logSm = torch.nn.functional.log_softmax (logProbs, -1L)
    let targets = torch.zeros ([| 4L; 1L |], dtype = torch.int64, device = torch.CPU)

    let loss = Loss.nll logSm targets
    loss.shape |> should equal [||]

[<Fact>]
let ``Binary cross-entropy with logit returns scalar`` () =
    let inp = torch.randn ([| 8L; 1L |], dtype = torch.float32, device = torch.CPU)
    let target = torch.zeros ([| 8L; 1L |], dtype = torch.float32, device = torch.CPU)

    let loss = Loss.binaryCrossEntropyWithLogit inp target
    loss.shape |> should equal [||]
    let v = loss.ToSingle()
    v |> should be (greaterThan 0.0f)

[<Fact>]
let ``L1 loss returns scalar`` () =
    let inp = torch.randn ([| 4L; 3L |], dtype = torch.float32, device = torch.CPU)
    let target = torch.randn ([| 4L; 3L |], dtype = torch.float32, device = torch.CPU)

    let loss = Loss.l1 inp target
    loss.shape |> should equal [||]
    let v = loss.ToSingle()
    v |> should be (greaterThan 0.0f)

[<Fact>]
let ``L1 loss of identical tensors is zero`` () =
    let t = torch.ones ([| 2L; 3L |], dtype = torch.float32, device = torch.CPU)

    let loss = Loss.l1 t t
    let v = loss.ToSingle()
    v |> should be (lessThan 1e-6f)

[<Fact>]
let ``Smooth L1 loss returns scalar`` () =
    let inp = torch.randn ([| 4L; 3L |], dtype = torch.float32, device = torch.CPU)
    let target = torch.randn ([| 4L; 3L |], dtype = torch.float32, device = torch.CPU)

    let loss = Loss.smoothL1 1.0 inp target
    loss.shape |> should equal [||]
    let v = loss.ToSingle()
    v |> should be (greaterThan 0.0f)

[<Fact>]
let ``Smooth L1 loss of identical tensors is zero`` () =
    let t = torch.ones ([| 2L; 3L |], dtype = torch.float32, device = torch.CPU)

    let loss = Loss.smoothL1 1.0 t t
    let v = loss.ToSingle()
    v |> should be (lessThan 1e-6f)

[<Fact>]
let ``KL divergence loss returns scalar`` () =
    let logProbs = torch.randn ([| 4L; 3L |], dtype = torch.float32, device = torch.CPU)
    let inp = torch.nn.functional.log_softmax (logProbs, -1L)
    let target = torch.nn.functional.softmax (logProbs, -1L)

    let loss = Loss.klDiv inp target
    loss.shape |> should equal [||]

[<Fact>]
let ``CoSENT loss is zero when all labels are equal`` () =
    let scores =
        torch.tensor ([| 0.9f; 0.1f; 0.5f |], dtype = torch.float32, device = torch.CPU)

    let labels =
        torch.tensor ([| 1.0f; 1.0f; 1.0f |], dtype = torch.float32, device = torch.CPU)

    let loss = Loss.cosent 20.0 scores labels
    loss.shape |> should equal [||]
    loss.ToSingle() |> should (equalWithin 1e-5f) 0.0f

[<Fact>]
let ``CoSENT loss is higher when ranking contradicts labels`` () =
    let labels =
        torch.tensor ([| 1.0f; 0.0f |], dtype = torch.float32, device = torch.CPU)

    let correct =
        torch.tensor ([| 0.9f; 0.1f |], dtype = torch.float32, device = torch.CPU)

    let inverted =
        torch.tensor ([| 0.1f; 0.9f |], dtype = torch.float32, device = torch.CPU)

    let correctLoss = Loss.cosent 20.0 correct labels
    let invertedLoss = Loss.cosent 20.0 inverted labels

    invertedLoss.ToSingle()
    |> should be (greaterThan (correctLoss.ToSingle()))
