module VarTests

open Xunit
open FsUnit.Xunit
open Toro
open Toro.NN
open TestHelper

// --- Model.namedParams tests ---

[<Fact>]
let ``namedParams collects tensors from Linear`` () =
    let linear = Linear.init 4 2 F32 Cpu
    let ps = Model.namedParams linear
    ps.Length |> should equal 2
    ps |> List.map fst |> should contain "Weight"
    ps |> List.map fst |> should contain "Bias"

[<Fact>]
let ``namedParams skips None bias`` () =
    let linear = Linear.initNoBias 4 2 F32 Cpu
    let ps = Model.namedParams linear
    ps.Length |> should equal 1
    ps |> List.map fst |> should equal [ "Weight" ]

[<Fact>]
let ``namedParams recurses into nested records`` () =
    let linear = Linear.init 4 2 F32 Cpu

    let model = {|
        L1 = linear
        L2 = Linear.init 2 1 F32 Cpu
    |}

    let ps = Model.namedParams model
    ps.Length |> should equal 4
    ps |> List.map fst |> should contain "L1.Weight"
    ps |> List.map fst |> should contain "L1.Bias"
    ps |> List.map fst |> should contain "L2.Weight"
    ps |> List.map fst |> should contain "L2.Bias"

[<Fact>]
let ``trainableVars returns only requiresGrad tensors`` () =
    let bn = BatchNorm.initDefault 4 F32 Cpu
    let all = Model.namedParams bn
    let trainable = Model.trainableVars bn

    all.Length |> should be (greaterThan trainable.Length)

    trainable
    |> List.iter (fun t -> t.RequiresGrad |> should be True)

// --- Model.save / loadInto tests ---

[<Fact>]
let ``Model save and loadInto round-trips`` () =
    let dir =
        System.IO.Path.Combine(System.IO.Path.GetTempPath(), System.Guid.NewGuid().ToString())

    try
        System.IO.Directory.CreateDirectory dir |> ignore
        let path = System.IO.Path.Combine(dir, "model.safetensors")

        let linear = Linear.init 3 2 F32 Cpu

        let wSum = (linear.Weight.sumAll ()).toFloat32Scalar ()


        Model.save linear path

        let linear2 = Linear.init 3 2 F32 Cpu
        let report = Model.loadInto linear2 path Strict

        report.Loaded.Length |> should equal 2
        report.Missing |> should be Empty
        report.Unexpected |> should be Empty
        report.ShapeMismatches |> should be Empty
        report.DTypeMismatches |> should be Empty

        let wSum2 = (linear2.Weight.sumAll ()).toFloat32Scalar ()


        wSum2 |> should (equalWithin 1e-5f) wSum
    finally
        if System.IO.Directory.Exists dir then
            System.IO.Directory.Delete(dir, true)

[<Fact>]
let ``Model loadInto Lenient reports shape mismatch`` () =
    let dir =
        System.IO.Path.Combine(System.IO.Path.GetTempPath(), System.Guid.NewGuid().ToString())

    try
        System.IO.Directory.CreateDirectory dir |> ignore
        let path = System.IO.Path.Combine(dir, "model.safetensors")

        let linear = Linear.init 3 2 F32 Cpu
        Model.save linear path

        let linear2 = Linear.init 5 2 F32 Cpu
        let report = Model.loadInto linear2 path Lenient

        report.ShapeMismatches.Length |> should equal 1
        report.ShapeMismatches[0].Name |> should equal "Weight"
        report.Loaded |> should contain "Bias"
    finally
        if System.IO.Directory.Exists dir then
            System.IO.Directory.Delete(dir, true)

[<Fact>]
let ``Model loadInto loads only required tensors`` () =
    let dir =
        System.IO.Path.Combine(System.IO.Path.GetTempPath(), System.Guid.NewGuid().ToString())

    try
        System.IO.Directory.CreateDirectory dir |> ignore
        let path = System.IO.Path.Combine(dir, "model.safetensors")

        let t1 = Tensor.randn ([ 1; 3 ], F32, Cpu)
        let t2 = Tensor.randn ([ 1 ], F32, Cpu)
        let t3 = Tensor.randn ([ 5 ], F32, Cpu)

        SafeTensors.save (Map [ "Weight", t1; "Bias", t2; "Extra", t3 ]) path


        let linear = Linear.init 3 1 F32 Cpu
        let report = Model.loadInto linear path Lenient

        report.Unexpected |> should contain "Extra"
        report.Loaded.Length |> should equal 2
    finally
        if System.IO.Directory.Exists dir then
            System.IO.Directory.Delete(dir, true)

// --- Init tests ---

[<Fact>]
let ``Uniform init produces values in range`` () =
    let lo, up = -1.0, 1.0

    let t = Init.toTensor [ 10000 ] F64 Cpu (Init.Uniform(lo, up))


    let mean = (t.meanAll ()).toFloat64Scalar ()

    mean |> should be (greaterThan (lo + 0.1))
    mean |> should be (lessThan (up - 0.1))

[<Fact>]
let ``KaimingNormal init has reasonable variance`` () =
    let shape = [ 256; 128 ]
    let fanIn = 128

    let t = Init.toTensor shape F32 Cpu Init.KaimingNormal

    let expectedStd = sqrt (2.0 / float fanIn)

    let mean = (t.meanAll ()).toFloat32Scalar ()

    let sqr = t.sqr ()
    let meanSqr = (sqr.meanAll ()).toFloat32Scalar ()
    let variance = float meanSqr - (float mean * float mean)
    let actualStd = sqrt variance

    abs (actualStd - expectedStd) |> should be (lessThan 0.02)

[<Fact>]
let ``Init Const creates tensor with given value`` () =
    let t = Init.toTensor [ 3; 2 ] F32 Cpu (Init.Const 5.0)

    t.Shape |> should equal [ 3; 2 ]
    let sum = (t.sumAll ()).toFloat32Scalar ()
    sum |> should equal 30.0f

[<Fact>]
let ``Init Randn creates tensor with specified mean`` () =
    let t = Init.toTensor [ 10000 ] F64 Cpu (Init.Randn(3.0, 0.01))


    let mean = (t.meanAll ()).toFloat64Scalar ()
    mean |> should (equalWithin 0.1) 3.0
