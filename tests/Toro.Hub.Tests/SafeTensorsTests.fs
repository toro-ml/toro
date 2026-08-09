module SafeTensorsTests

open System.IO
open System.Text
open Xunit
open FsUnit.Xunit
open Toro
open Toro.NN
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
    let report = Model.loadFromDict linear2 dict None Strict |> unwrap

    report.Loaded.Length |> should equal 2
    report.Missing |> should be Empty
    report.Unexpected |> should be Empty
    report.ShapeMismatches |> should be Empty
    report.DTypeMismatches |> should be Empty

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

    let report =
        Model.loadFromDict linear2 renamedDict (Some nameMap) Strict
        |> unwrap

    report.Loaded.Length |> should equal 2

    let w1 = Model.namedParams linear |> List.head |> snd |> tensorSum
    let w2 = Model.namedParams linear2 |> List.head |> snd |> tensorSum
    w2 |> should (equalWithin 1e-5f) w1

[<Fact>]
let ``Model.loadFromDict Lenient reports missing keys`` () =
    let linear = Linear.init 4 2 F32 Cpu |> unwrap
    let before = Model.namedParams linear |> List.head |> snd |> tensorSum
    let report = Model.loadFromDict linear Map.empty None Lenient |> unwrap
    let after = Model.namedParams linear |> List.head |> snd |> tensorSum
    after |> should (equalWithin 1e-5f) before
    report.Loaded |> should be Empty
    report.Missing.Length |> should equal 2
    report.Missing |> should contain "Weight"
    report.Missing |> should contain "Bias"

[<Fact>]
let ``Model.loadFromDict Strict fails on missing keys`` () =
    let linear = Linear.init 4 2 F32 Cpu |> unwrap

    match Model.loadFromDict linear Map.empty None Strict with
    | Error _ -> ()
    | Ok _ -> failwith "Expected Error for missing keys in Strict mode"

[<Fact>]
let ``Model.loadFromDict Strict fails on unexpected keys`` () =
    let linear = Linear.init 4 2 F32 Cpu |> unwrap
    let original = Model.namedParams linear

    let dict =
        original
        |> List.map (fun (name, t) -> name, t)
        |> Map.ofList
        |> Map.add "Extra" (Tensor.randn ([ 2 ], F32, Cpu) |> unwrap)

    let linear2 = Linear.init 4 2 F32 Cpu |> unwrap

    match Model.loadFromDict linear2 dict None Strict with
    | Error _ -> ()
    | Ok _ -> failwith "Expected Error for unexpected keys in Strict mode"

// --- Validation and mismatch tests ---

[<Fact>]
let ``SafeTensors loadMeta returns tensor metadata`` () =
    withTempDir (fun dir ->
        let path = Path.Combine(dir, "meta.safetensors")
        let t1 = Tensor.randn ([ 3; 4 ], F32, Cpu) |> unwrap
        let t2 = Tensor.randn ([ 5 ], F64, Cpu) |> unwrap
        let tensors = Map [ "weight", t1; "bias", t2 ]

        SafeTensors.save tensors path |> unwrap
        let meta = SafeTensors.loadMeta path |> unwrap

        meta |> Map.count |> should equal 2
        meta["weight"].DType |> should equal F32
        meta["weight"].Shape |> should equal [ 3; 4 ]
        meta["bias"].DType |> should equal F64
        meta["bias"].Shape |> should equal [ 5 ])

[<Fact>]
let ``SafeTensors save produces 8-byte aligned header`` () =
    withTempDir (fun dir ->
        let path = Path.Combine(dir, "aligned.safetensors")
        let t = Tensor.randn ([ 2 ], F32, Cpu) |> unwrap
        SafeTensors.save (Map [ "x", t ]) path |> unwrap

        use fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read)
        use reader = new BinaryReader(fs)
        let headerSize = reader.ReadUInt64() |> int
        headerSize % 8 |> should equal 0)

[<Fact>]
let ``SafeTensors loadSelected loads only requested tensors`` () =
    withTempDir (fun dir ->
        let path = Path.Combine(dir, "selective.safetensors")
        let t1 = Tensor.randn ([ 3 ], F32, Cpu) |> unwrap
        let t2 = Tensor.randn ([ 5 ], F32, Cpu) |> unwrap
        let t3 = Tensor.randn ([ 7 ], F32, Cpu) |> unwrap
        let tensors = Map [ "a", t1; "b", t2; "c", t3 ]

        SafeTensors.save tensors path |> unwrap
        let meta, loaded = SafeTensors.loadSelected path (Set [ "a"; "c" ]) |> unwrap

        meta |> Map.count |> should equal 3
        loaded |> Map.count |> should equal 2
        loaded |> Map.containsKey "a" |> should equal true
        loaded |> Map.containsKey "c" |> should equal true
        loaded |> Map.containsKey "b" |> should equal false)

[<Fact>]
let ``Model.loadFromDict Lenient reports shape mismatch`` () =
    let linear = Linear.init 4 2 F32 Cpu |> unwrap

    let wrongShapeDict =
        Map [
            "Weight", Tensor.randn ([ 3; 2 ], F32, Cpu) |> unwrap
            "Bias", Tensor.randn ([ 2 ], F32, Cpu) |> unwrap
        ]

    let report =
        Model.loadFromDict linear wrongShapeDict None Lenient
        |> unwrap

    report.ShapeMismatches.Length |> should equal 1
    report.ShapeMismatches[0].Name |> should equal "Weight"
    report.Loaded |> should contain "Bias"

[<Fact>]
let ``Model.loadFromDict Lenient reports dtype mismatch`` () =
    let linear = Linear.init 4 2 F32 Cpu |> unwrap

    let wrongDTypeDict =
        Map [
            "Weight", Tensor.randn ([ 2; 4 ], F64, Cpu) |> unwrap
            "Bias", Tensor.randn ([ 2 ], F32, Cpu) |> unwrap
        ]

    let report =
        Model.loadFromDict linear wrongDTypeDict None Lenient
        |> unwrap

    report.DTypeMismatches.Length |> should equal 1
    report.DTypeMismatches[0].Name |> should equal "Weight"
    report.Loaded |> should contain "Bias"

[<Fact>]
let ``Model.loadFromDict Strict fails on shape mismatch`` () =
    let linear = Linear.init 4 2 F32 Cpu |> unwrap

    let wrongShapeDict =
        Map [
            "Weight", Tensor.randn ([ 3; 2 ], F32, Cpu) |> unwrap
            "Bias", Tensor.randn ([ 2 ], F32, Cpu) |> unwrap
        ]

    match Model.loadFromDict linear wrongShapeDict None Strict with
    | Error _ -> ()
    | Ok _ -> failwith "Expected Error for shape mismatch in Strict mode"

[<Fact>]
let ``Model.loadFromDict Strict fails on dtype mismatch`` () =
    let linear = Linear.init 4 2 F32 Cpu |> unwrap

    let wrongDTypeDict =
        Map [
            "Weight", Tensor.randn ([ 2; 4 ], F64, Cpu) |> unwrap
            "Bias", Tensor.randn ([ 2 ], F32, Cpu) |> unwrap
        ]

    match Model.loadFromDict linear wrongDTypeDict None Strict with
    | Error _ -> ()
    | Ok _ -> failwith "Expected Error for dtype mismatch in Strict mode"

// --- Spec-fidelity tests ---

let private writeSafeTensorsRaw (path: string) (headerJson: string) (data: byte array) =
    use fs = new FileStream(path, FileMode.Create, FileAccess.Write)
    use bw = new BinaryWriter(fs)
    let headerBytes = Encoding.UTF8.GetBytes(headerJson)
    bw.Write(uint64 headerBytes.Length)
    bw.Write(headerBytes)
    bw.Write(data)

[<Fact>]
let ``SafeTensors rejects negative shape dimension`` () =
    withTempDir (fun dir ->
        let path = Path.Combine(dir, "neg.safetensors")

        let header = """{"bad":{"dtype":"F32","shape":[-1,4],"data_offsets":[0,16]}}"""

        writeSafeTensorsRaw path header (Array.zeroCreate 16)

        match SafeTensors.load path with
        | Error(Msg msg) -> msg |> should haveSubstring "negative dimension"
        | Error e -> failwith $"Expected Msg error, got: %A{e}"
        | Ok _ -> failwith "Expected Error for negative dimension")

[<Fact>]
let ``SafeTensors allows zero-element tensor`` () =
    withTempDir (fun dir ->
        let path = Path.Combine(dir, "zero.safetensors")

        let header = """{"empty":{"dtype":"F32","shape":[0,4],"data_offsets":[0,0]}}"""

        writeSafeTensorsRaw path header Array.empty

        let loaded = SafeTensors.load path |> unwrap
        loaded |> Map.count |> should equal 1
        loaded["empty"].Shape |> should equal [ 0; 4 ])

[<Fact>]
let ``SafeTensors rejects offset gap`` () =
    withTempDir (fun dir ->
        let path = Path.Combine(dir, "gap.safetensors")

        let header =
            """{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[8,12]}}"""

        writeSafeTensorsRaw path header (Array.zeroCreate 12)

        match SafeTensors.load path with
        | Error(Msg msg) -> msg |> should haveSubstring "offset gap"
        | Error e -> failwith $"Expected Msg error, got: %A{e}"
        | Ok _ -> failwith "Expected Error for offset gap")

[<Fact>]
let ``SafeTensors rejects overlapping offsets`` () =
    withTempDir (fun dir ->
        let path = Path.Combine(dir, "overlap.safetensors")

        let header =
            """{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"""

        writeSafeTensorsRaw path header (Array.zeroCreate 8)

        match SafeTensors.load path with
        | Error(Msg msg) -> msg |> should haveSubstring "offset gap"
        | Error e -> failwith $"Expected Msg error, got: %A{e}"
        | Ok _ -> failwith "Expected Error for overlapping offsets")

[<Fact>]
let ``SafeTensors save orders by descending dtype alignment then name`` () =
    withTempDir (fun dir ->
        let path = Path.Combine(dir, "sorted.safetensors")
        let u8 = Tensor.zeros ([ 1 ], U8, Cpu) |> unwrap
        let f64 = Tensor.randn ([ 1 ], F64, Cpu) |> unwrap
        let f32 = Tensor.randn ([ 1 ], F32, Cpu) |> unwrap

        SafeTensors.save (Map [ "z_u8", u8; "a_f32", f32; "b_f64", f64 ]) path
        |> unwrap

        let meta = SafeTensors.loadMeta path |> unwrap

        let ordered =
            meta
            |> Map.toList
            |> List.sortBy (fun (_, m) -> m.StartOffset)

        let names = ordered |> List.map fst
        names |> should equal [ "b_f64"; "a_f32"; "z_u8" ])

[<Fact>]
let ``SafeTensors rejects header not starting with brace`` () =
    withTempDir (fun dir ->
        let path = Path.Combine(dir, "bad.safetensors")
        let badHeader = "[\"not an object\"]"
        writeSafeTensorsRaw path badHeader Array.empty

        match SafeTensors.load path with
        | Error(Msg msg) -> msg |> should haveSubstring "start with '{'"
        | Error e -> failwith $"Expected Msg error, got: %A{e}"
        | Ok _ -> failwith "Expected Error for invalid header")
