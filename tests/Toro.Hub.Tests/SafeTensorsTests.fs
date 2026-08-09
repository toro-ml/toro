module SafeTensorsTests

open System.IO
open Xunit
open FsUnit.Xunit
open Toro
open Toro.NN
open Toro.Hub
open TestHelper

let private withTempDir f =
    let dir = Path.Combine(Path.GetTempPath(), $"toro-st-{System.Guid.NewGuid()}")

    try
        Directory.CreateDirectory dir |> ignore
        f dir
    finally
        if Directory.Exists dir then
            Directory.Delete(dir, true)

let private tensorSum (t: Tensor) =
    (t.sumAll () |> unwrap).toFloat32Scalar () |> unwrap

[<Fact>]
let ``SafeTensors round-trip F32`` () =
    withTempDir (fun dir ->
        let path = Path.Combine(dir, "test.safetensors")
        let t1 = Tensor.randn ([ 3; 4 ], F32, Cpu) |> unwrap
        let t2 = Tensor.randn ([ 5 ], F32, Cpu) |> unwrap
        let tensors = Map [ "weight", t1; "bias", t2 ]

        SafeTensors.save tensors path |> unwrap
        let loaded = SafeTensors.load path |> unwrap

        loaded |> Map.count |> should equal 2
        loaded["weight"].Shape |> should equal [ 3; 4 ]
        loaded["bias"].Shape |> should equal [ 5 ]

        tensorSum loaded["weight"]
        |> should (equalWithin 1e-5f) (tensorSum t1)

        tensorSum loaded["bias"]
        |> should (equalWithin 1e-5f) (tensorSum t2))

[<Fact>]
let ``SafeTensors round-trip I64`` () =
    withTempDir (fun dir ->
        let path = Path.Combine(dir, "int.safetensors")

        let t = Tensor.ofArray ([| 1L; 2L; 3L; 4L; 5L; 6L |], Cpu) |> unwrap

        let tensors = Map [ "ids", t ]

        SafeTensors.save tensors path |> unwrap
        let loaded = SafeTensors.load path |> unwrap

        loaded["ids"].Shape |> should equal [ 6 ]

        let orig = (t.sumAll () |> unwrap).toInt64Scalar () |> unwrap

        let roundTripped =
            (loaded["ids"].sumAll () |> unwrap).toInt64Scalar ()
            |> unwrap

        roundTripped |> should equal orig)

[<Fact>]
let ``SafeTensors round-trip multiple dtypes`` () =
    withTempDir (fun dir ->
        let path = Path.Combine(dir, "multi.safetensors")
        let f32 = Tensor.randn ([ 2; 3 ], F32, Cpu) |> unwrap
        let f64 = Tensor.randn ([ 4 ], F64, Cpu) |> unwrap
        let tensors = Map [ "f32", f32; "f64", f64 ]

        SafeTensors.save tensors path |> unwrap
        let loaded = SafeTensors.load path |> unwrap

        loaded["f32"].DType |> should equal F32
        loaded["f64"].DType |> should equal F64)

[<Fact>]
let ``SafeTensors empty map produces valid file`` () =
    withTempDir (fun dir ->
        let path = Path.Combine(dir, "empty.safetensors")
        SafeTensors.save Map.empty path |> unwrap
        let loaded = SafeTensors.load path |> unwrap
        loaded |> Map.isEmpty |> should equal true)

[<Fact>]
let ``SafeTensors scalar tensor round-trip`` () =
    withTempDir (fun dir ->
        let path = Path.Combine(dir, "scalar.safetensors")
        let t = Tensor.ofArray ([| 42.0f |], Cpu) |> unwrap
        let tensors = Map [ "val", t ]

        SafeTensors.save tensors path |> unwrap
        let loaded = SafeTensors.load path |> unwrap

        tensorSum loaded["val"] |> should (equalWithin 1e-5f) 42.0f)

[<Fact>]
let ``Model.loadFromDict without nameMap`` () =
    let linear = Linear.init 4 2 F32 Cpu |> unwrap
    let original = Model.namedParams linear

    let dict =
        original
        |> List.map (fun (name, t) -> name, t)
        |> Map.ofList

    let linear2 = Linear.init 4 2 F32 Cpu |> unwrap
    Model.loadFromDict linear2 dict None |> unwrap

    let w1 = Model.namedParams linear |> List.head |> snd |> tensorSum
    let w2 = Model.namedParams linear2 |> List.head |> snd |> tensorSum
    w2 |> should (equalWithin 1e-5f) w1

[<Fact>]
let ``Model.loadFromDict with nameMap`` () =
    let linear = Linear.init 4 2 F32 Cpu |> unwrap
    let original = Model.namedParams linear

    let renamedDict =
        original
        |> List.map (fun (name, t) -> "hf." + name, t)
        |> Map.ofList

    let nameMap =
        original
        |> List.map (fun (name, _) -> "hf." + name, name)
        |> Map.ofList

    let linear2 = Linear.init 4 2 F32 Cpu |> unwrap

    Model.loadFromDict linear2 renamedDict (Some nameMap)
    |> unwrap

    let w1 = Model.namedParams linear |> List.head |> snd |> tensorSum
    let w2 = Model.namedParams linear2 |> List.head |> snd |> tensorSum
    w2 |> should (equalWithin 1e-5f) w1

[<Fact>]
let ``Model.loadFromDict ignores missing keys`` () =
    let linear = Linear.init 4 2 F32 Cpu |> unwrap
    let before = Model.namedParams linear |> List.head |> snd |> tensorSum
    Model.loadFromDict linear Map.empty None |> unwrap
    let after = Model.namedParams linear |> List.head |> snd |> tensorSum
    after |> should (equalWithin 1e-5f) before
