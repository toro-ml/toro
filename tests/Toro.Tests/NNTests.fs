module NNTests

open Xunit
open FsUnit.Xunit
open Toro
open Toro.NN

let unwrap r =
    match r with
    | Ok v -> v
    | Error e -> failwithf "Unexpected error: %A" e

[<Fact>]
let ``Linear forward produces correct shape`` () =
    let vb = VarBuilder.fromInit F32 Cpu

    let linear = Linear.create 10 5 vb |> unwrap

    let x = Tensor.randn ([ 2; 10 ], F32, Cpu) |> unwrap

    let y = linear.forward x |> unwrap
    y.Shape |> should equal [ 2; 5 ]

[<Fact>]
let ``Linear no bias forward works`` () =
    let vb = VarBuilder.fromInit F32 Cpu

    let linear = Linear.createNoBias 8 4 vb |> unwrap

    let x = Tensor.randn ([ 3; 8 ], F32, Cpu) |> unwrap

    let y = linear.forward x |> unwrap
    y.Shape |> should equal [ 3; 4 ]
    linear.Bias |> should equal None

[<Fact>]
let ``Embedding forward produces correct shape`` () =
    let vb = VarBuilder.fromInit F32 Cpu

    let emb = Embedding.create 100 16 vb |> unwrap

    let ids = Tensor.zeros ([ 2; 5 ], I64, Cpu) |> unwrap

    let y = emb.forward ids |> unwrap
    y.Shape |> should equal [ 2; 5; 16 ]

[<Fact>]
let ``LayerNorm forward preserves shape`` () =
    let vb = VarBuilder.fromInit F32 Cpu

    let ln = LayerNorm.createDefault 8 vb |> unwrap

    let x = Tensor.randn ([ 2; 3; 8 ], F32, Cpu) |> unwrap

    let y = ln.forward x |> unwrap
    y.Shape |> should equal [ 2; 3; 8 ]

[<Fact>]
let ``RmsNorm forward preserves shape`` () =
    let vb = VarBuilder.fromInit F32 Cpu
    let rms = RmsNorm.create 8 1e-5 vb |> unwrap

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
    let vb = VarBuilder.fromInit F32 Cpu

    let model =
        result {
            let! l1 = Linear.create 10 5 (vb |> VarBuilder.pp "l1")

            let! l2 = Linear.create 5 2 (vb |> VarBuilder.pp "l2")

            return
                Sequential.create [
                    l1 :> IModule
                    Relu :> IModule
                    l2 :> IModule
                ]
        }
        |> unwrap

    let x = Tensor.randn ([ 4; 10 ], F32, Cpu) |> unwrap

    let y = model.forward x |> unwrap
    y.Shape |> should equal [ 4; 2 ]

[<Fact>]
let ``VarBuilder pp creates namespaced keys`` () =
    let t = Tensor.ones ([ 3; 2 ], F32, Cpu) |> unwrap

    let tensors = Map.ofList [ "layer.weight", t ]

    let vb = VarBuilder.fromTensors tensors F32 Cpu

    let vbLayer = vb |> VarBuilder.pp "layer"

    VarBuilder.containsTensor "weight" vbLayer |> should be True

[<Fact>]
let ``VarBuilder returns error on missing tensor`` () =
    let tensors = Map.empty

    let vb = VarBuilder.fromTensors tensors F32 Cpu

    let r = VarBuilder.get [ 2; 3 ] "missing" vb

    match r with
    | Error(TensorNotFound _) -> ()
    | _ -> failwith "Expected TensorNotFound"

[<Fact>]
let ``VarBuilder shape mismatch returns ShapeMismatch`` () =
    let t = Tensor.ones ([ 3; 2 ], F32, Cpu) |> unwrap
    let tensors = Map.ofList [ "w", t ]
    let vb = VarBuilder.fromTensors tensors F32 Cpu

    match VarBuilder.get [ 4; 2 ] "w" vb with
    | Error(ShapeMismatch _) -> ()
    | other -> failwithf "Expected ShapeMismatch, got: %A" other

[<Fact>]
let ``Uniform init produces values in range`` () =
    let lo, up = -1.0, 1.0

    let t =
        Init.toTensor [ 10000 ] F64 Cpu (Init.Uniform(lo, up))
        |> unwrap

    let mean = (t.meanAll () |> unwrap).toFloat64Scalar () |> unwrap

    mean |> should be (greaterThan (lo + 0.1))
    mean |> should be (lessThan (up - 0.1))

[<Fact>]
let ``KaimingNormal init has reasonable variance`` () =
    let shape = [ 256; 128 ]
    let fanIn = 128

    let t = Init.toTensor shape F32 Cpu Init.KaimingNormal |> unwrap

    let expectedStd = sqrt (2.0 / float fanIn)

    let mean = (t.meanAll () |> unwrap).toFloat32Scalar () |> unwrap

    let sqr = t.sqr () |> unwrap
    let meanSqr = (sqr.meanAll () |> unwrap).toFloat32Scalar () |> unwrap
    let variance = float meanSqr - (float mean * float mean)
    let actualStd = sqrt variance

    abs (actualStd - expectedStd) |> should be (lessThan 0.02)

// --- VarMap tests ---

[<Fact>]
let ``VarMap get-or-create returns same tensor`` () =
    let vm = VarMap()
    let t1 = vm.get [ 3; 2 ] "w" Init.KaimingNormal F32 Cpu |> unwrap
    let t2 = vm.get [ 3; 2 ] "w" Init.KaimingNormal F32 Cpu |> unwrap
    obj.ReferenceEquals(t1.Inner, t2.Inner) |> should be True

[<Fact>]
let ``VarMap allVars returns all registered tensors`` () =
    let vm = VarMap()
    vm.get [ 2; 3 ] "a" (Init.Const 1.0) F32 Cpu |> unwrap |> ignore
    vm.get [ 4 ] "b" (Init.Const 0.0) F32 Cpu |> unwrap |> ignore
    vm.allVars().Length |> should equal 2

[<Fact>]
let ``VarMap shape mismatch returns error`` () =
    let vm = VarMap()
    vm.get [ 3; 2 ] "w" Init.KaimingNormal F32 Cpu |> unwrap |> ignore

    match vm.get [ 4; 2 ] "w" Init.KaimingNormal F32 Cpu with
    | Error(ShapeMismatch _) -> ()
    | other -> failwithf "Expected ShapeMismatch, got: %A" other

[<Fact>]
let ``VarBuilder fromVarMap creates trainable params`` () =
    let vm = VarMap()
    let vb = VarBuilder.fromVarMap vm F32 Cpu

    let linear = Linear.create 4 2 (vb |> VarBuilder.pp "l") |> unwrap

    let vars = vm.allVars ()
    vars.Length |> should equal 2

    let x = Tensor.randn ([ 1; 4 ], F32, Cpu) |> unwrap
    let y = linear.forward x |> unwrap
    y.Shape |> should equal [ 1; 2 ]

// --- Loss tests ---

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

// --- SGD tests ---

[<Fact>]
let ``SGD step reduces loss`` () =
    let vm = VarMap()
    let vb = VarBuilder.fromVarMap vm F32 Cpu

    let linear = Linear.create 4 2 (vb |> VarBuilder.pp "l") |> unwrap

    let x = Tensor.randn ([ 8; 4 ], F32, Cpu) |> unwrap
    let target = Tensor.randn ([ 8; 2 ], F32, Cpu) |> unwrap

    let opt = SGD.create 0.01 (vm.allVars ()) :> IOptimizer

    let getLoss () =
        result {
            let! y = linear.forward x
            return! Loss.mse y target
        }
        |> unwrap

    let loss0 = (getLoss ()).toFloat32Scalar () |> unwrap

    for _ in 1..20 do
        let loss = getLoss ()
        opt.backwardStep loss |> unwrap

    let lossN = (getLoss ()).toFloat32Scalar () |> unwrap
    lossN |> should be (lessThan loss0)

// --- AdamW tests ---

[<Fact>]
let ``AdamW step reduces loss`` () =
    let vm = VarMap()
    let vb = VarBuilder.fromVarMap vm F32 Cpu

    let linear = Linear.create 4 2 (vb |> VarBuilder.pp "l") |> unwrap

    let x = Tensor.randn ([ 8; 4 ], F32, Cpu) |> unwrap
    let target = Tensor.randn ([ 8; 2 ], F32, Cpu) |> unwrap

    let opt =
        AdamW.createWithLr 0.01 (vm.allVars ()) |> unwrap :> IOptimizer

    let getLoss () =
        result {
            let! y = linear.forward x
            return! Loss.mse y target
        }
        |> unwrap

    let loss0 = (getLoss ()).toFloat32Scalar () |> unwrap

    for _ in 1..20 do
        let loss = getLoss ()
        opt.backwardStep loss |> unwrap

    let lossN = (getLoss ()).toFloat32Scalar () |> unwrap
    lossN |> should be (lessThan loss0)
