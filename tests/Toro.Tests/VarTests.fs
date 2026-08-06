module VarTests

open Xunit
open FsUnit.Xunit
open Toro
open Toro.NN
open TestHelper

// --- VarBuilder tests ---

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
let ``VarBuilder fromVarMap creates trainable params`` () =
    let vm = VarMap()
    let vb = VarBuilder.fromVarMap vm F32 Cpu

    let linear = Linear.create 4 2 (vb |> VarBuilder.pp "l") |> unwrap

    let vars = vm.allVars ()
    vars.Length |> should equal 2

    let x = Tensor.randn ([ 1; 4 ], F32, Cpu) |> unwrap
    let y = linear.forward x |> unwrap
    y.Shape |> should equal [ 1; 2 ]

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

// --- Init tests ---

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

[<Fact>]
let ``Init Const creates tensor with given value`` () =
    let t = Init.toTensor [ 3; 2 ] F32 Cpu (Init.Const 5.0) |> unwrap

    t.Shape |> should equal [ 3; 2 ]
    let sum = (t.sumAll () |> unwrap).toFloat32Scalar () |> unwrap
    sum |> should equal 30.0f

[<Fact>]
let ``Init Randn creates tensor with specified mean`` () =
    let t = Init.toTensor [ 10000 ] F64 Cpu (Init.Randn(3.0, 0.01)) |> unwrap

    let mean = (t.meanAll () |> unwrap).toFloat64Scalar () |> unwrap
    mean |> should (equalWithin 0.1) 3.0

// --- VarMap save/load tests ---

[<Fact>]
let ``VarMap save and load round-trips`` () =
    let dir = System.IO.Path.Combine(System.IO.Path.GetTempPath(), System.Guid.NewGuid().ToString())

    try
        let vm = VarMap()
        vm.get [ 3; 2 ] "w" (Init.Const 1.0) F32 Cpu |> unwrap |> ignore
        vm.get [ 4 ] "b" (Init.Const 2.0) F32 Cpu |> unwrap |> ignore

        vm.save dir |> unwrap

        let vm2 = VarMap.load dir |> unwrap
        let data = vm2.data ()
        data.Count |> should equal 2
        data.ContainsKey "w" |> should be True
        data.ContainsKey "b" |> should be True

        let wShape = data["w"].Shape
        wShape |> should equal [ 3; 2 ]

        let bShape = data["b"].Shape
        bShape |> should equal [ 4 ]
    finally
        if System.IO.Directory.Exists dir then
            System.IO.Directory.Delete(dir, true)
