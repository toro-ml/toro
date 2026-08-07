module LayerTests

open Xunit
open FsUnit.Xunit
open Toro
open Toro.NN
open TestHelper

[<Fact>]
let ``Linear forward produces correct shape`` () =
    let linear = Linear.init 10 5 F32 Cpu |> unwrap

    let x = Tensor.randn ([ 2; 10 ], F32, Cpu) |> unwrap

    let y = linear.forward x |> unwrap
    y.Shape |> should equal [ 2; 5 ]

[<Fact>]
let ``Linear no bias forward works`` () =
    let linear = Linear.initNoBias 8 4 F32 Cpu |> unwrap

    let x = Tensor.randn ([ 3; 8 ], F32, Cpu) |> unwrap

    let y = linear.forward x |> unwrap
    y.Shape |> should equal [ 3; 4 ]
    linear.Bias |> should equal None

[<Fact>]
let ``Embedding forward produces correct shape`` () =
    let emb = Embedding.init 100 16 F32 Cpu |> unwrap

    let ids = Tensor.zeros ([ 2; 5 ], I64, Cpu) |> unwrap

    let y = emb.forward ids |> unwrap
    y.Shape |> should equal [ 2; 5; 16 ]

[<Fact>]
let ``LayerNorm forward preserves shape`` () =
    let ln = LayerNorm.initDefault 8 F32 Cpu |> unwrap

    let x = Tensor.randn ([ 2; 3; 8 ], F32, Cpu) |> unwrap

    let y = ln.forward x |> unwrap
    y.Shape |> should equal [ 2; 3; 8 ]

[<Fact>]
let ``RmsNorm forward preserves shape`` () =
    let rms = RmsNorm.init 8 1e-5 F32 Cpu |> unwrap

    let x = Tensor.randn ([ 2; 3; 8 ], F32, Cpu) |> unwrap

    let y = rms.forward x |> unwrap
    y.Shape |> should equal [ 2; 3; 8 ]

[<Fact>]
let ``Activation functions produce same shape`` () =
    let x = Tensor.randn ([ 2; 4 ], F32, Cpu) |> unwrap

    for act in [ Relu; Gelu; Silu; Tanh; Sigmoid ] do
        let y = act.forward x |> unwrap
        y.Shape |> should equal x.Shape

[<Fact>]
let ``Sequential chains modules`` () =
    let model =
        result {
            let! l1 = Linear.init 10 5 F32 Cpu
            let! l2 = Linear.init 5 2 F32 Cpu

            return Sequential.create [ l1 :> IModule; Relu :> IModule; l2 :> IModule ]
        }
        |> unwrap

    let x = Tensor.randn ([ 4; 10 ], F32, Cpu) |> unwrap

    let y = model.forward x |> unwrap
    y.Shape |> should equal [ 4; 2 ]

[<Fact>]
let ``ModuleT ofModule delegates to IModule forward`` () =
    let linear = Linear.init 4 2 F32 Cpu |> unwrap

    let moduleT = ModuleT.ofModule (linear :> IModule)

    let x = Tensor.randn ([ 1; 4 ], F32, Cpu) |> unwrap
    let y = moduleT.forwardT x true |> unwrap
    y.Shape |> should equal [ 1; 2 ]

    let y2 = moduleT.forwardT x false |> unwrap
    y2.Shape |> should equal [ 1; 2 ]

[<Fact>]
let ``Dropout train=false passes input through`` () =
    let drop = Dropout.create 0.5
    let x = Tensor.ones ([ 4; 8 ], F32, Cpu) |> unwrap

    let y = drop.forwardT x false |> unwrap
    let sum = (y.sumAll () |> unwrap).toFloat32Scalar () |> unwrap
    sum |> should equal 32.0f

[<Fact>]
let ``Dropout train=true produces zeros`` () =
    let drop = Dropout.create 0.5
    let x = Tensor.ones ([ 100; 100 ], F32, Cpu) |> unwrap

    let y = drop.forwardT x true |> unwrap
    let sum = (y.sumAll () |> unwrap).toFloat32Scalar () |> unwrap

    let sumSq =
        ((y.sqr () |> unwrap).sumAll () |> unwrap).toFloat32Scalar ()
        |> unwrap

    sumSq |> should be (greaterThan 15000.0f)
    sum |> should be (greaterThan 0.0f)

[<Fact>]
let ``Dropout with dropP=0 passes input through even in train`` () =
    let drop = Dropout.create 0.0
    let x = Tensor.ones ([ 4; 8 ], F32, Cpu) |> unwrap

    let y = drop.forwardT x true |> unwrap
    let sum = (y.sumAll () |> unwrap).toFloat32Scalar () |> unwrap
    sum |> should equal 32.0f

[<Fact>]
let ``Dropout implements IModuleT`` () =
    let drop = Dropout.create 0.3 :> IModuleT
    let x = Tensor.randn ([ 2; 4 ], F32, Cpu) |> unwrap
    let y = drop.forwardT x false |> unwrap
    y.Shape |> should equal [ 2; 4 ]

[<Fact>]
let ``Func create wraps function as IModule`` () =
    let f = Func.create (fun x -> x.relu ())
    let m = f :> IModule
    let x = Tensor.full ([ 3 ], -1.0, F32, Cpu) |> unwrap

    let y = m.forward x |> unwrap
    let sum = (y.sumAll () |> unwrap).toFloat32Scalar () |> unwrap
    sum |> should equal 0.0f

[<Fact>]
let ``FuncT passes train flag correctly`` () =
    let f = FuncT.create (fun x train -> if train then x.mulScalar 2.0 else Ok x)

    let m = f :> IModuleT
    let x = Tensor.ones ([ 3 ], F32, Cpu) |> unwrap

    let yTrain = m.forwardT x true |> unwrap

    (yTrain.sumAll () |> unwrap).toFloat32Scalar ()
    |> unwrap
    |> should equal 6.0f

    let yEval = m.forwardT x false |> unwrap

    (yEval.sumAll () |> unwrap).toFloat32Scalar ()
    |> unwrap
    |> should equal 3.0f

[<Fact>]
let ``Identity returns input unchanged`` () =
    let m = Identity() :> IModule
    let x = Tensor.randn ([ 2; 3 ], F32, Cpu) |> unwrap

    let y = m.forward x |> unwrap
    y.Shape |> should equal [ 2; 3 ]

    let diff = (x.sub y |> unwrap).sumAll () |> unwrap
    let v = diff.toFloat32Scalar () |> unwrap
    v |> should equal 0.0f

// --- Conv1d tests ---

[<Fact>]
let ``Conv1d forward produces correct shape`` () =
    let conv = Conv1d.initDefault 3 16 5 F32 Cpu |> unwrap

    let x = Tensor.randn ([ 1; 3; 20 ], F32, Cpu) |> unwrap
    let y = conv.forward x |> unwrap
    y.Shape |> should equal [ 1; 16; 16 ]

[<Fact>]
let ``Conv1d with padding preserves length`` () =
    let config = {
        Conv1dConfig.defaultConfig with
            Padding = 2
    }

    let conv = Conv1d.init 1 8 5 config F32 Cpu |> unwrap

    let x = Tensor.randn ([ 1; 1; 10 ], F32, Cpu) |> unwrap
    let y = conv.forward x |> unwrap
    y.Shape |> should equal [ 1; 8; 10 ]

[<Fact>]
let ``Conv1d no bias works`` () =
    let conv =
        Conv1d.initNoBias 2 4 3 Conv1dConfig.defaultConfig F32 Cpu
        |> unwrap

    conv.Bias |> should equal None
    let x = Tensor.randn ([ 1; 2; 8 ], F32, Cpu) |> unwrap
    let y = conv.forward x |> unwrap
    y.Shape |> should equal [ 1; 4; 6 ]

// --- Conv2d tests ---

[<Fact>]
let ``Conv2d forward produces correct shape`` () =
    let conv = Conv2d.initDefault 1 8 3 F32 Cpu |> unwrap

    let x = Tensor.randn ([ 1; 1; 8; 8 ], F32, Cpu) |> unwrap
    let y = conv.forward x |> unwrap
    y.Shape |> should equal [ 1; 8; 6; 6 ]

[<Fact>]
let ``Conv2d with padding preserves spatial dims`` () =
    let config = {
        Conv2dConfig.defaultConfig with
            Padding = 1
    }

    let conv = Conv2d.init 1 4 3 config F32 Cpu |> unwrap

    let x = Tensor.randn ([ 2; 1; 6; 6 ], F32, Cpu) |> unwrap
    let y = conv.forward x |> unwrap
    y.Shape |> should equal [ 2; 4; 6; 6 ]

[<Fact>]
let ``Conv2d no bias works`` () =
    let conv =
        Conv2d.initNoBias 3 16 3 Conv2dConfig.defaultConfig F32 Cpu
        |> unwrap

    conv.Bias |> should equal None
    let x = Tensor.randn ([ 1; 3; 10; 10 ], F32, Cpu) |> unwrap
    let y = conv.forward x |> unwrap
    y.Shape |> should equal [ 1; 16; 8; 8 ]

// --- BatchNorm tests ---

[<Fact>]
let ``BatchNorm forward preserves shape in eval mode`` () =
    let bn = BatchNorm.initDefault 8 F32 Cpu |> unwrap

    let x = Tensor.randn ([ 2; 8; 4; 4 ], F32, Cpu) |> unwrap
    let y = bn.forwardT x false |> unwrap
    y.Shape |> should equal [ 2; 8; 4; 4 ]

[<Fact>]
let ``BatchNorm forward preserves shape in train mode`` () =
    let bn = BatchNorm.initDefault 4 F32 Cpu |> unwrap

    let x = Tensor.randn ([ 4; 4; 3; 3 ], F32, Cpu) |> unwrap
    let y = bn.forwardT x true |> unwrap
    y.Shape |> should equal [ 4; 4; 3; 3 ]

[<Fact>]
let ``BatchNorm implements IModuleT`` () =
    let bn = BatchNorm.initDefault 4 F32 Cpu |> unwrap :> IModuleT

    let x = Tensor.randn ([ 2; 4; 3; 3 ], F32, Cpu) |> unwrap
    let y = bn.forwardT x false |> unwrap
    y.Shape |> should equal [ 2; 4; 3; 3 ]

// --- GroupNorm tests ---

[<Fact>]
let ``GroupNorm forward preserves shape`` () =
    let gn = GroupNorm.initDefault 4 8 F32 Cpu |> unwrap

    let x = Tensor.randn ([ 2; 8; 4; 4 ], F32, Cpu) |> unwrap
    let y = gn.forward x |> unwrap
    y.Shape |> should equal [ 2; 8; 4; 4 ]

[<Fact>]
let ``GroupNorm implements IModule`` () =
    let gn = GroupNorm.initDefault 2 4 F32 Cpu |> unwrap :> IModule

    let x = Tensor.randn ([ 1; 4; 6; 6 ], F32, Cpu) |> unwrap
    let y = gn.forward x |> unwrap
    y.Shape |> should equal [ 1; 4; 6; 6 ]

// --- Activation extension tests ---

[<Fact>]
let ``LeakyRelu activation produces correct shape`` () =
    let act = LeakyRelu 0.01 :> IModule
    let x = Tensor.randn ([ 3; 4 ], F32, Cpu) |> unwrap
    let y = act.forward x |> unwrap
    y.Shape |> should equal [ 3; 4 ]

[<Fact>]
let ``Elu activation produces correct shape`` () =
    let act = Elu 1.0 :> IModule
    let x = Tensor.randn ([ 3; 4 ], F32, Cpu) |> unwrap
    let y = act.forward x |> unwrap
    y.Shape |> should equal [ 3; 4 ]

[<Fact>]
let ``Mish activation produces correct shape`` () =
    let act = Mish :> IModule
    let x = Tensor.randn ([ 3; 4 ], F32, Cpu) |> unwrap
    let y = act.forward x |> unwrap
    y.Shape |> should equal [ 3; 4 ]

// --- SequentialT tests ---

[<Fact>]
let ``SequentialT chains IModuleT layers`` () =
    let linear = Linear.init 10 5 F32 Cpu |> unwrap

    let seq =
        SequentialT.create [
            ModuleT.ofModule (linear :> IModule)
            ModuleT.ofModule (Relu :> IModule)
            Dropout.create 0.0 :> IModuleT
        ]

    let x = Tensor.randn ([ 2; 10 ], F32, Cpu) |> unwrap
    let y = seq.forwardT x true |> unwrap
    y.Shape |> should equal [ 2; 5 ]

[<Fact>]
let ``SequentialT ofModules wraps IModule list`` () =
    let linear = Linear.init 10 5 F32 Cpu |> unwrap

    let seq = SequentialT.ofModules [ linear :> IModule; Relu :> IModule ]

    let x = Tensor.randn ([ 2; 10 ], F32, Cpu) |> unwrap
    let y = seq.forwardT x false |> unwrap
    y.Shape |> should equal [ 2; 5 ]

[<Fact>]
let ``SequentialT implements IModuleT`` () =
    let seq = SequentialT.create [ ModuleT.ofModule (Relu :> IModule) ] :> IModuleT

    let x = Tensor.randn ([ 3; 4 ], F32, Cpu) |> unwrap
    let y = seq.forwardT x false |> unwrap
    y.Shape |> should equal [ 3; 4 ]
