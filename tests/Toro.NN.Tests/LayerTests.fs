module LayerTests

open Xunit
open FsUnit.Xunit
open Toro
open Toro.NN
open TestHelper

[<Fact>]
let ``Linear forward produces correct shape`` () =
    let linear = Linear.init 10 5 F32 Cpu

    let x = Tensor.randn ([ 2; 10 ], F32, Cpu)

    let y = linear.forward x
    y.Shape |> should equal [ 2; 5 ]

[<Fact>]
let ``Linear no bias forward works`` () =
    let linear = Linear.initNoBias 8 4 F32 Cpu

    let x = Tensor.randn ([ 3; 8 ], F32, Cpu)

    let y = linear.forward x
    y.Shape |> should equal [ 3; 4 ]
    linear.Bias |> should equal None

[<Fact>]
let ``Embedding forward produces correct shape`` () =
    let emb = Embedding.init 100 16 F32 Cpu

    let ids = Tensor.zeros ([ 2; 5 ], I64, Cpu)

    let y = emb.forward ids
    y.Shape |> should equal [ 2; 5; 16 ]

[<Fact>]
let ``LayerNorm forward preserves shape`` () =
    let ln = LayerNorm.initDefault 8 F32 Cpu

    let x = Tensor.randn ([ 2; 3; 8 ], F32, Cpu)

    let y = ln.forward x
    y.Shape |> should equal [ 2; 3; 8 ]

[<Fact>]
let ``RmsNorm forward preserves shape`` () =
    let rms = RmsNorm.init 8 1e-5 F32 Cpu

    let x = Tensor.randn ([ 2; 3; 8 ], F32, Cpu)

    let y = rms.forward x
    y.Shape |> should equal [ 2; 3; 8 ]

[<Fact>]
let ``Activation functions produce same shape`` () =
    let x = Tensor.randn ([ 2; 4 ], F32, Cpu)

    for act in [ Relu; Gelu; Silu; Tanh; Sigmoid ] do
        let y = act.forward x
        y.Shape |> should equal x.Shape

[<Fact>]
let ``Sequential chains modules`` () =
    let l1 = Linear.init 10 5 F32 Cpu
    let l2 = Linear.init 5 2 F32 Cpu

    let model =
        sequential {
            l1
            Relu
            l2
        }

    let x = Tensor.randn ([ 4; 10 ], F32, Cpu)

    let y = model.forward x
    y.Shape |> should equal [ 4; 2 ]

[<Fact>]
let ``Dropout train=false passes input through`` () =
    let drop = Dropout.create 0.5
    let x = Tensor.ones ([ 4; 8 ], F32, Cpu)

    let y = drop.forwardT false x
    let sum = (y.sumAll ()).toFloat32Scalar ()
    sum |> should equal 32.0f

[<Fact>]
let ``Dropout train=true produces zeros`` () =
    let drop = Dropout.create 0.5
    let x = Tensor.ones ([ 100; 100 ], F32, Cpu)

    let y = drop.forwardT true x
    let sum = (y.sumAll ()).toFloat32Scalar ()

    let sumSq = ((y.sqr ()).sumAll ()).toFloat32Scalar ()


    sumSq |> should be (greaterThan 15000.0f)
    sum |> should be (greaterThan 0.0f)

[<Fact>]
let ``Dropout with dropP=0 passes input through even in train`` () =
    let drop = Dropout.create 0.0
    let x = Tensor.ones ([ 4; 8 ], F32, Cpu)

    let y = drop.forwardT true x
    let sum = (y.sumAll ()).toFloat32Scalar ()
    sum |> should equal 32.0f

[<Fact>]
let ``Func create wraps function as IModule`` () =
    let f = Func.create _.relu()
    let m = f :> IModule
    let x = Tensor.full ([ 3 ], -1.0, F32, Cpu)

    let y = m.forward x
    let sum = (y.sumAll ()).toFloat32Scalar ()
    sum |> should equal 0.0f

[<Fact>]
let ``Identity returns input unchanged`` () =
    let m = Func.Identity
    let x = Tensor.randn ([ 2; 3 ], F32, Cpu)

    let y = m.forward x
    y.Shape |> should equal [ 2; 3 ]

    let diff = (x.sub y).sumAll ()
    let v = diff.toFloat32Scalar ()
    v |> should equal 0.0f

// --- Conv1d tests ---

[<Fact>]
let ``Conv1d forward produces correct shape`` () =
    let conv = Conv1d.initDefault 3 16 5 F32 Cpu

    let x = Tensor.randn ([ 1; 3; 20 ], F32, Cpu)
    let y = conv.forward x
    y.Shape |> should equal [ 1; 16; 16 ]

[<Fact>]
let ``Conv1d with padding preserves length`` () =
    let config = {
        Conv1dConfig.defaultConfig with
            Padding = 2
    }

    let conv = Conv1d.init 1 8 5 config F32 Cpu

    let x = Tensor.randn ([ 1; 1; 10 ], F32, Cpu)
    let y = conv.forward x
    y.Shape |> should equal [ 1; 8; 10 ]

[<Fact>]
let ``Conv1d no bias works`` () =
    let conv = Conv1d.initNoBias 2 4 3 Conv1dConfig.defaultConfig F32 Cpu


    conv.Bias |> should equal None
    let x = Tensor.randn ([ 1; 2; 8 ], F32, Cpu)
    let y = conv.forward x
    y.Shape |> should equal [ 1; 4; 6 ]

// --- Conv2d tests ---

[<Fact>]
let ``Conv2d forward produces correct shape`` () =
    let conv = Conv2d.initDefault 1 8 3 F32 Cpu

    let x = Tensor.randn ([ 1; 1; 8; 8 ], F32, Cpu)
    let y = conv.forward x
    y.Shape |> should equal [ 1; 8; 6; 6 ]

[<Fact>]
let ``Conv2d with padding preserves spatial dims`` () =
    let config = {
        Conv2dConfig.defaultConfig with
            Padding = 1
    }

    let conv = Conv2d.init 1 4 3 config F32 Cpu

    let x = Tensor.randn ([ 2; 1; 6; 6 ], F32, Cpu)
    let y = conv.forward x
    y.Shape |> should equal [ 2; 4; 6; 6 ]

[<Fact>]
let ``Conv2d no bias works`` () =
    let conv = Conv2d.initNoBias 3 16 3 Conv2dConfig.defaultConfig F32 Cpu


    conv.Bias |> should equal None
    let x = Tensor.randn ([ 1; 3; 10; 10 ], F32, Cpu)
    let y = conv.forward x
    y.Shape |> should equal [ 1; 16; 8; 8 ]

// --- BatchNorm tests ---

[<Fact>]
let ``BatchNorm forward preserves shape in eval mode`` () =
    let bn = BatchNorm.initDefault 8 F32 Cpu

    let x = Tensor.randn ([ 2; 8; 4; 4 ], F32, Cpu)
    let y = bn.forwardT false x
    y.Shape |> should equal [ 2; 8; 4; 4 ]

[<Fact>]
let ``BatchNorm forward preserves shape in train mode`` () =
    let bn = BatchNorm.initDefault 4 F32 Cpu

    let x = Tensor.randn ([ 4; 4; 3; 3 ], F32, Cpu)
    let y = bn.forwardT true x
    y.Shape |> should equal [ 4; 4; 3; 3 ]

// --- GroupNorm tests ---

[<Fact>]
let ``GroupNorm forward preserves shape`` () =
    let gn = GroupNorm.initDefault 4 8 F32 Cpu

    let x = Tensor.randn ([ 2; 8; 4; 4 ], F32, Cpu)
    let y = gn.forward x
    y.Shape |> should equal [ 2; 8; 4; 4 ]

[<Fact>]
let ``GroupNorm implements IModule`` () =
    let gn = GroupNorm.initDefault 2 4 F32 Cpu :> IModule

    let x = Tensor.randn ([ 1; 4; 6; 6 ], F32, Cpu)
    let y = gn.forward x
    y.Shape |> should equal [ 1; 4; 6; 6 ]

// --- Activation extension tests ---

[<Fact>]
let ``LeakyRelu activation produces correct shape`` () =
    let act = LeakyRelu 0.01 :> IModule
    let x = Tensor.randn ([ 3; 4 ], F32, Cpu)
    let y = act.forward x
    y.Shape |> should equal [ 3; 4 ]

[<Fact>]
let ``Elu activation produces correct shape`` () =
    let act = Elu 1.0 :> IModule
    let x = Tensor.randn ([ 3; 4 ], F32, Cpu)
    let y = act.forward x
    y.Shape |> should equal [ 3; 4 ]

[<Fact>]
let ``Mish activation produces correct shape`` () =
    let act = Mish :> IModule
    let x = Tensor.randn ([ 3; 4 ], F32, Cpu)
    let y = act.forward x
    y.Shape |> should equal [ 3; 4 ]

// --- pipeline CE tests ---

[<Fact>]
let ``pipeline composes IModule layers`` () =
    let linear = Linear.init 10 5 F32 Cpu

    let f =
        pipeline {
            linear
            Relu
        }

    let x = Tensor.randn ([ 2; 10 ], F32, Cpu)
    let y = f x
    y.Shape |> should equal [ 2; 5 ]

[<Fact>]
let ``pipeline composes functions and modules`` () =
    let linear = Linear.init 10 5 F32 Cpu
    let drop = Dropout.create 0.0

    let f =
        pipeline {
            linear
            fun (x: Tensor) -> x.relu ()
            drop.forwardT false
        }

    let x = Tensor.randn ([ 2; 10 ], F32, Cpu)
    let y = f x
    y.Shape |> should equal [ 2; 5 ]

[<Fact>]
let ``pipeline propagates exceptions`` () =
    let fail = Func.create (fun _ -> failwith "test error")

    let f =
        pipeline {
            fail
            Relu
        }

    let x = Tensor.randn ([ 2; 3 ], F32, Cpu)

    try
        f x |> ignore
        failwith "Expected exception"
    with
    | :? System.Exception as ex when ex.Message = "test error" -> ()
    | ex -> Assert.Fail $"Expected exception with message \"test error\", got %A{ex}"
