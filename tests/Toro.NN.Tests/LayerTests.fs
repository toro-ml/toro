module LayerTests

open Xunit
open FsUnit.Xunit
open Toro
open TorchSharp
open Toro.NN
open TestHelper

[<Fact>]
let ``Linear forward produces correct shape`` () =
    let linear = Linear.init 10 5 torch.float32 torch.CPU

    let x = torch.randn ([| 2L; 10L |], dtype = torch.float32, device = torch.CPU)

    let y = linear.forward x
    y.shape |> should equal [| 2L; 5L |]

[<Fact>]
let ``Linear no bias forward works`` () =
    let linear = Linear.initNoBias 8 4 torch.float32 torch.CPU

    let x = torch.randn ([| 3L; 8L |], dtype = torch.float32, device = torch.CPU)

    let y = linear.forward x
    y.shape |> should equal [| 3L; 4L |]
    linear.Bias |> should equal None

[<Fact>]
let ``Embedding forward produces correct shape`` () =
    let emb = Embedding.init 100 16 torch.float32 torch.CPU

    let ids = torch.zeros ([| 2L; 5L |], dtype = torch.int64, device = torch.CPU)

    let y = emb.forward ids
    y.shape |> should equal [| 2L; 5L; 16L |]

[<Fact>]
let ``LayerNorm forward preserves shape`` () =
    let ln = LayerNorm.initDefault 8 torch.float32 torch.CPU

    let x = torch.randn ([| 2L; 3L; 8L |], dtype = torch.float32, device = torch.CPU)

    let y = ln.forward x
    y.shape |> should equal [| 2L; 3L; 8L |]

[<Fact>]
let ``RmsNorm forward preserves shape`` () =
    let rms = RmsNorm.init 8 1e-5 torch.float32 torch.CPU

    let x = torch.randn ([| 2L; 3L; 8L |], dtype = torch.float32, device = torch.CPU)

    let y = rms.forward x
    y.shape |> should equal [| 2L; 3L; 8L |]

[<Fact>]
let ``Activation functions produce same shape`` () =
    let x = torch.randn ([| 2L; 4L |], dtype = torch.float32, device = torch.CPU)

    for act in [ Relu; Gelu; Silu; Tanh; Sigmoid ] do
        let y = act.forward x
        y.shape |> should equal x.shape

[<Fact>]
let ``Sequential chains modules`` () =
    let l1 = Linear.init 10 5 torch.float32 torch.CPU
    let l2 = Linear.init 5 2 torch.float32 torch.CPU

    let model =
        sequential {
            l1
            Relu
            l2
        }

    let x = torch.randn ([| 4L; 10L |], dtype = torch.float32, device = torch.CPU)

    let y = model.forward x
    y.shape |> should equal [| 4L; 2L |]

[<Fact>]
let ``Dropout train=false passes input through`` () =
    let drop = Dropout.create 0.5
    let x = torch.ones ([| 4L; 8L |], dtype = torch.float32, device = torch.CPU)

    let y = drop.forwardT false x
    let sum = (y.sum ()).ToSingle()
    sum |> should equal 32.0f

[<Fact>]
let ``Dropout train=true produces zeros`` () =
    let drop = Dropout.create 0.5
    let x = torch.ones ([| 100L; 100L |], dtype = torch.float32, device = torch.CPU)

    let y = drop.forwardT true x
    let sum = (y.sum ()).ToSingle()

    let sumSq = ((y.square ()).sum ()).ToSingle()


    sumSq |> should be (greaterThan 15000.0f)
    sum |> should be (greaterThan 0.0f)

[<Fact>]
let ``Dropout with dropP=0 passes input through even in train`` () =
    let drop = Dropout.create 0.0
    let x = torch.ones ([| 4L; 8L |], dtype = torch.float32, device = torch.CPU)

    let y = drop.forwardT true x
    let sum = (y.sum ()).ToSingle()
    sum |> should equal 32.0f

[<Fact>]
let ``Func create wraps function as IModule`` () =
    let f = Func.create _.relu()
    let m = f :> IModule

    let x =
        torch.full ([| 3L |], scalar -1.0, dtype = torch.float32, device = torch.CPU)

    let y = m.forward x
    let sum = (y.sum ()).ToSingle()
    sum |> should equal 0.0f

[<Fact>]
let ``Identity returns input unchanged`` () =
    let m = Func.Identity
    let x = torch.randn ([| 2L; 3L |], dtype = torch.float32, device = torch.CPU)

    let y = m.forward x
    y.shape |> should equal [| 2L; 3L |]

    let diff = (x.sub y).sum ()
    let v = diff.ToSingle()
    v |> should equal 0.0f

// --- Conv1d tests ---

[<Fact>]
let ``Conv1d forward produces correct shape`` () =
    let conv = Conv1d.initDefault 3 16 5 torch.float32 torch.CPU

    let x = torch.randn ([| 1L; 3L; 20L |], dtype = torch.float32, device = torch.CPU)
    let y = conv.forward x
    y.shape |> should equal [| 1L; 16L; 16L |]

[<Fact>]
let ``Conv1d with padding preserves length`` () =
    let config = {
        Conv1dConfig.defaultConfig with
            Padding = 2
    }

    let conv = Conv1d.init 1 8 5 config torch.float32 torch.CPU

    let x = torch.randn ([| 1L; 1L; 10L |], dtype = torch.float32, device = torch.CPU)
    let y = conv.forward x
    y.shape |> should equal [| 1L; 8L; 10L |]

[<Fact>]
let ``Conv1d no bias works`` () =
    let conv =
        Conv1d.initNoBias 2 4 3 Conv1dConfig.defaultConfig torch.float32 torch.CPU


    conv.Bias |> should equal None
    let x = torch.randn ([| 1L; 2L; 8L |], dtype = torch.float32, device = torch.CPU)
    let y = conv.forward x
    y.shape |> should equal [| 1L; 4L; 6L |]

// --- Conv2d tests ---

[<Fact>]
let ``Conv2d forward produces correct shape`` () =
    let conv = Conv2d.initDefault 1 8 3 torch.float32 torch.CPU

    let x =
        torch.randn ([| 1L; 1L; 8L; 8L |], dtype = torch.float32, device = torch.CPU)

    let y = conv.forward x
    y.shape |> should equal [| 1L; 8L; 6L; 6L |]

[<Fact>]
let ``Conv2d with padding preserves spatial dims`` () =
    let config = {
        Conv2dConfig.defaultConfig with
            Padding = 1
    }

    let conv = Conv2d.init 1 4 3 config torch.float32 torch.CPU

    let x =
        torch.randn ([| 2L; 1L; 6L; 6L |], dtype = torch.float32, device = torch.CPU)

    let y = conv.forward x
    y.shape |> should equal [| 2L; 4L; 6L; 6L |]

[<Fact>]
let ``Conv2d no bias works`` () =
    let conv =
        Conv2d.initNoBias 3 16 3 Conv2dConfig.defaultConfig torch.float32 torch.CPU


    conv.Bias |> should equal None

    let x =
        torch.randn ([| 1L; 3L; 10L; 10L |], dtype = torch.float32, device = torch.CPU)

    let y = conv.forward x
    y.shape |> should equal [| 1L; 16L; 8L; 8L |]

// --- BatchNorm tests ---

[<Fact>]
let ``BatchNorm forward preserves shape in eval mode`` () =
    let bn = BatchNorm.initDefault 8 torch.float32 torch.CPU

    let x =
        torch.randn ([| 2L; 8L; 4L; 4L |], dtype = torch.float32, device = torch.CPU)

    let y = bn.forwardT false x
    y.shape |> should equal [| 2L; 8L; 4L; 4L |]

[<Fact>]
let ``BatchNorm forward preserves shape in train mode`` () =
    let bn = BatchNorm.initDefault 4 torch.float32 torch.CPU

    let x =
        torch.randn ([| 4L; 4L; 3L; 3L |], dtype = torch.float32, device = torch.CPU)

    let y = bn.forwardT true x
    y.shape |> should equal [| 4L; 4L; 3L; 3L |]

// --- GroupNorm tests ---

[<Fact>]
let ``GroupNorm forward preserves shape`` () =
    let gn = GroupNorm.initDefault 4 8 torch.float32 torch.CPU

    let x =
        torch.randn ([| 2L; 8L; 4L; 4L |], dtype = torch.float32, device = torch.CPU)

    let y = gn.forward x
    y.shape |> should equal [| 2L; 8L; 4L; 4L |]

[<Fact>]
let ``GroupNorm implements IModule`` () =
    let gn = GroupNorm.initDefault 2 4 torch.float32 torch.CPU :> IModule

    let x =
        torch.randn ([| 1L; 4L; 6L; 6L |], dtype = torch.float32, device = torch.CPU)

    let y = gn.forward x
    y.shape |> should equal [| 1L; 4L; 6L; 6L |]

// --- Activation extension tests ---

[<Fact>]
let ``LeakyRelu activation produces correct shape`` () =
    let act = LeakyRelu 0.01 :> IModule
    let x = torch.randn ([| 3L; 4L |], dtype = torch.float32, device = torch.CPU)
    let y = act.forward x
    y.shape |> should equal [| 3L; 4L |]

[<Fact>]
let ``Elu activation produces correct shape`` () =
    let act = Elu 1.0 :> IModule
    let x = torch.randn ([| 3L; 4L |], dtype = torch.float32, device = torch.CPU)
    let y = act.forward x
    y.shape |> should equal [| 3L; 4L |]

[<Fact>]
let ``Mish activation produces correct shape`` () =
    let act = Mish :> IModule
    let x = torch.randn ([| 3L; 4L |], dtype = torch.float32, device = torch.CPU)
    let y = act.forward x
    y.shape |> should equal [| 3L; 4L |]

// --- pipeline CE tests ---

[<Fact>]
let ``pipeline composes IModule layers`` () =
    let linear = Linear.init 10 5 torch.float32 torch.CPU

    let f =
        pipeline {
            linear
            Relu
        }

    let x = torch.randn ([| 2L; 10L |], dtype = torch.float32, device = torch.CPU)
    let y = f x
    y.shape |> should equal [| 2L; 5L |]

[<Fact>]
let ``pipeline composes functions and modules`` () =
    let linear = Linear.init 10 5 torch.float32 torch.CPU
    let drop = Dropout.create 0.0

    let f =
        pipeline {
            linear
            fun (x: Tensor) -> x.relu ()
            drop.forwardT false
        }

    let x = torch.randn ([| 2L; 10L |], dtype = torch.float32, device = torch.CPU)
    let y = f x
    y.shape |> should equal [| 2L; 5L |]

[<Fact>]
let ``pipeline propagates exceptions`` () =
    let fail = Func.create (fun _ -> failwith "test error")

    let f =
        pipeline {
            fail
            Relu
        }

    let x = torch.randn ([| 2L; 3L |], dtype = torch.float32, device = torch.CPU)

    try
        f x |> ignore
        failwith "Expected exception"
    with
    | ex when ex.Message = "test error" -> ()
    | ex -> Assert.Fail $"Expected exception with message \"test error\", got %A{ex}"
