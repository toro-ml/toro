module SafeTensorsTests

open System.IO
open System.Text
open Xunit
open FsUnit.Xunit
open Toro
open TorchSharp
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

let private tensorSum (t: Tensor) = (t.sum ()).ToSingle()

[<Fact>]
let ``SafeTensors round-trip torch.float32`` () =
    withTempDir (fun dir ->
        let path = Path.Combine(dir, "test.safetensors")
        let t1 = torch.randn ([| 3L; 4L |], dtype = torch.float32, device = torch.CPU)
        let t2 = torch.randn ([| 5L |], dtype = torch.float32, device = torch.CPU)
        let tensors = Map [ "weight", t1; "bias", t2 ]

        SafeTensors.save tensors path
        let loaded = SafeTensors.load path

        loaded |> Map.count |> should equal 2
        loaded["weight"].shape |> should equal [| 3L; 4L |]
        loaded["bias"].shape |> should equal [| 5L |]

        tensorSum loaded["weight"]
        |> should (equalWithin 1e-5f) (tensorSum t1)

        tensorSum loaded["bias"]
        |> should (equalWithin 1e-5f) (tensorSum t2))

[<Fact>]
let ``SafeTensors round-trip torch.int64`` () =
    withTempDir (fun dir ->
        let path = Path.Combine(dir, "int.safetensors")

        let t = torch.tensor ([| 1L; 2L; 3L; 4L; 5L; 6L |], device = torch.CPU)

        let tensors = Map [ "ids", t ]

        SafeTensors.save tensors path
        let loaded = SafeTensors.load path

        loaded["ids"].shape |> should equal [| 6L |]

        let orig = (t.sum ()).ToInt64()

        let roundTripped = (loaded["ids"].sum ()).ToInt64()


        roundTripped |> should equal orig)

[<Fact>]
let ``SafeTensors round-trip multiple dtypes`` () =
    withTempDir (fun dir ->
        let path = Path.Combine(dir, "multi.safetensors")
        let f32 = torch.randn ([| 2L; 3L |], dtype = torch.float32, device = torch.CPU)
        let f64 = torch.randn ([| 4L |], dtype = torch.float64, device = torch.CPU)
        let tensors = Map [ "f32", f32; "f64", f64 ]

        SafeTensors.save tensors path
        let loaded = SafeTensors.load path

        loaded["f32"].dtype |> should equal torch.float32
        loaded["f64"].dtype |> should equal torch.float64)

[<Fact>]
let ``SafeTensors empty map produces valid file`` () =
    withTempDir (fun dir ->
        let path = Path.Combine(dir, "empty.safetensors")
        SafeTensors.save Map.empty path
        let loaded = SafeTensors.load path
        loaded |> Map.isEmpty |> should equal true)

[<Fact>]
let ``SafeTensors scalar tensor round-trip`` () =
    withTempDir (fun dir ->
        let path = Path.Combine(dir, "scalar.safetensors")
        let t = torch.tensor ([| 42.0f |], device = torch.CPU)
        let tensors = Map [ "val", t ]

        SafeTensors.save tensors path
        let loaded = SafeTensors.load path

        tensorSum loaded["val"] |> should (equalWithin 1e-5f) 42.0f)

[<Fact>]
let ``Model.loadFromDict without nameMap`` () =
    let linear = Linear.init 4 2 torch.float32 torch.CPU
    let original = Model.namedParams linear

    let dict =
        original
        |> List.map (fun item -> item.Name, item.Tensor)
        |> Map.ofList

    let linear2 = Linear.init 4 2 torch.float32 torch.CPU
    let report = Model.loadFromDict linear2 dict None Strict

    report.Loaded.Length |> should equal 2
    report.Missing |> should be Empty
    report.Unexpected |> should be Empty
    report.ShapeMismatches |> should be Empty
    report.DTypeMismatches |> should be Empty

    let w1 =
        Model.namedParams linear
        |> List.head
        |> _.Tensor
        |> tensorSum

    let w2 =
        Model.namedParams linear2
        |> List.head
        |> _.Tensor
        |> tensorSum

    w2 |> should (equalWithin 1e-5f) w1

[<Fact>]
let ``Model.loadFromDict with nameMap`` () =
    let linear = Linear.init 4 2 torch.float32 torch.CPU
    let original = Model.namedParams linear

    let renamedDict =
        original
        |> List.map (fun item -> "hf." + item.Name, item.Tensor)
        |> Map.ofList

    let nameMap =
        original
        |> List.map (fun item -> "hf." + item.Name, item.Name)
        |> Map.ofList

    let linear2 = Linear.init 4 2 torch.float32 torch.CPU

    let report = Model.loadFromDict linear2 renamedDict (Some nameMap) Strict


    report.Loaded.Length |> should equal 2

    let w1 =
        Model.namedParams linear
        |> List.head
        |> _.Tensor
        |> tensorSum

    let w2 =
        Model.namedParams linear2
        |> List.head
        |> _.Tensor
        |> tensorSum

    w2 |> should (equalWithin 1e-5f) w1

[<Fact>]
let ``Model.loadFromDict Lenient reports missing keys`` () =
    let linear = Linear.init 4 2 torch.float32 torch.CPU

    let before =
        Model.namedParams linear
        |> List.head
        |> _.Tensor
        |> tensorSum

    let report = Model.loadFromDict linear Map.empty None Lenient

    let after =
        Model.namedParams linear
        |> List.head
        |> _.Tensor
        |> tensorSum

    after |> should (equalWithin 1e-5f) before
    report.Loaded |> should be Empty
    report.Missing.Length |> should equal 2
    report.Missing |> should contain "Weight"
    report.Missing |> should contain "Bias"

[<Fact>]
let ``Model.loadFromDict Strict fails on missing keys`` () =
    let linear = Linear.init 4 2 torch.float32 torch.CPU

    try
        Model.loadFromDict linear Map.empty None Strict |> ignore
        failwith "Expected exception for missing keys in Strict mode"
    with _ ->
        ()

[<Fact>]
let ``Model.loadFromDict Strict fails on unexpected keys`` () =
    let linear = Linear.init 4 2 torch.float32 torch.CPU
    let original = Model.namedParams linear

    let dict =
        original
        |> List.map (fun item -> item.Name, item.Tensor)
        |> Map.ofList
        |> Map.add "Extra" (torch.randn ([| 2L |], dtype = torch.float32, device = torch.CPU))

    let linear2 = Linear.init 4 2 torch.float32 torch.CPU

    try
        Model.loadFromDict linear2 dict None Strict |> ignore
        failwith "Expected exception for unexpected keys in Strict mode"
    with _ ->
        ()

// --- Validation and mismatch tests ---

[<Fact>]
let ``SafeTensors loadMeta returns tensor metadata`` () =
    withTempDir (fun dir ->
        let path = Path.Combine(dir, "meta.safetensors")
        let t1 = torch.randn ([| 3L; 4L |], dtype = torch.float32, device = torch.CPU)
        let t2 = torch.randn ([| 5L |], dtype = torch.float64, device = torch.CPU)
        let tensors = Map [ "weight", t1; "bias", t2 ]

        SafeTensors.save tensors path
        let meta = SafeTensors.loadMeta path

        meta |> Map.count |> should equal 2
        meta["weight"].DType |> should equal torch.float32
        meta["weight"].Shape |> should equal [| 3L; 4L |]
        meta["bias"].DType |> should equal torch.float64
        meta["bias"].Shape |> should equal [| 5L |])

[<Fact>]
let ``SafeTensors save produces 8-byte aligned header`` () =
    withTempDir (fun dir ->
        let path = Path.Combine(dir, "aligned.safetensors")
        let t = torch.randn ([| 2L |], dtype = torch.float32, device = torch.CPU)
        SafeTensors.save (Map [ "x", t ]) path

        use fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read)
        use reader = new BinaryReader(fs)
        let headerSize = reader.ReadUInt64() |> int
        headerSize % 8 |> should equal 0)

[<Fact>]
let ``SafeTensors loadSelected loads only requested tensors`` () =
    withTempDir (fun dir ->
        let path = Path.Combine(dir, "selective.safetensors")
        let t1 = torch.randn ([| 3L |], dtype = torch.float32, device = torch.CPU)
        let t2 = torch.randn ([| 5L |], dtype = torch.float32, device = torch.CPU)
        let t3 = torch.randn ([| 7L |], dtype = torch.float32, device = torch.CPU)
        let tensors = Map [ "a", t1; "b", t2; "c", t3 ]

        SafeTensors.save tensors path
        let meta, loaded = SafeTensors.loadSelected path (Set [ "a"; "c" ])

        meta |> Map.count |> should equal 3
        loaded |> Map.count |> should equal 2
        loaded |> Map.containsKey "a" |> should equal true
        loaded |> Map.containsKey "c" |> should equal true
        loaded |> Map.containsKey "b" |> should equal false)

[<Fact>]
let ``Model.loadFromDict Lenient reports shape mismatch`` () =
    let linear = Linear.init 4 2 torch.float32 torch.CPU

    let wrongShapeDict =
        Map [
            "Weight", torch.randn ([| 3L; 2L |], dtype = torch.float32, device = torch.CPU)
            "Bias", torch.randn ([| 2L |], dtype = torch.float32, device = torch.CPU)
        ]

    let report = Model.loadFromDict linear wrongShapeDict None Lenient


    report.ShapeMismatches.Length |> should equal 1
    report.ShapeMismatches[0].Name |> should equal "Weight"
    report.Loaded |> should contain "Bias"

[<Fact>]
let ``Model.loadFromDict Lenient reports dtype mismatch`` () =
    let linear = Linear.init 4 2 torch.float32 torch.CPU

    let wrongDTypeDict =
        Map [
            "Weight", torch.randn ([| 2L; 4L |], dtype = torch.float64, device = torch.CPU)
            "Bias", torch.randn ([| 2L |], dtype = torch.float32, device = torch.CPU)
        ]

    let report = Model.loadFromDict linear wrongDTypeDict None Lenient


    report.DTypeMismatches.Length |> should equal 1
    report.DTypeMismatches[0].Name |> should equal "Weight"
    report.Loaded |> should contain "Bias"

[<Fact>]
let ``Model.loadFromDict Strict fails on shape mismatch`` () =
    let linear = Linear.init 4 2 torch.float32 torch.CPU

    let wrongShapeDict =
        Map [
            "Weight", torch.randn ([| 3L; 2L |], dtype = torch.float32, device = torch.CPU)
            "Bias", torch.randn ([| 2L |], dtype = torch.float32, device = torch.CPU)
        ]

    try
        Model.loadFromDict linear wrongShapeDict None Strict
        |> ignore

        failwith "Expected exception for shape mismatch in Strict mode"
    with _ ->
        ()

[<Fact>]
let ``Model.loadFromDict Strict fails on dtype mismatch`` () =
    let linear = Linear.init 4 2 torch.float32 torch.CPU

    let wrongDTypeDict =
        Map [
            "Weight", torch.randn ([| 2L; 4L |], dtype = torch.float64, device = torch.CPU)
            "Bias", torch.randn ([| 2L |], dtype = torch.float32, device = torch.CPU)
        ]

    try
        Model.loadFromDict linear wrongDTypeDict None Strict
        |> ignore

        failwith "Expected exception for dtype mismatch in Strict mode"
    with _ ->
        ()

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

        try
            SafeTensors.load path |> ignore
            failwith "Expected exception for negative dimension"
        with :? System.InvalidOperationException as ex ->
            ex.Message |> should haveSubstring "negative dimension")

[<Fact>]
let ``SafeTensors allows zero-element tensor`` () =
    withTempDir (fun dir ->
        let path = Path.Combine(dir, "zero.safetensors")

        let header = """{"empty":{"dtype":"F32","shape":[0,4],"data_offsets":[0,0]}}"""

        writeSafeTensorsRaw path header Array.empty

        let loaded = SafeTensors.load path
        loaded |> Map.count |> should equal 1
        loaded["empty"].shape |> should equal [| 0L; 4L |])

[<Fact>]
let ``SafeTensors rejects offset gap`` () =
    withTempDir (fun dir ->
        let path = Path.Combine(dir, "gap.safetensors")

        let header =
            """{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[8,12]}}"""

        writeSafeTensorsRaw path header (Array.zeroCreate 12)

        try
            SafeTensors.load path |> ignore
            failwith "Expected exception for offset gap"
        with :? System.InvalidOperationException as ex ->
            ex.Message |> should haveSubstring "offset gap")

[<Fact>]
let ``SafeTensors rejects overlapping offsets`` () =
    withTempDir (fun dir ->
        let path = Path.Combine(dir, "overlap.safetensors")

        let header =
            """{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"""

        writeSafeTensorsRaw path header (Array.zeroCreate 8)

        try
            SafeTensors.load path |> ignore
            failwith "Expected exception for overlapping offsets"
        with :? System.InvalidOperationException as ex ->
            ex.Message |> should haveSubstring "offset gap")

[<Fact>]
let ``SafeTensors save orders by descending dtype alignment then name`` () =
    withTempDir (fun dir ->
        let path = Path.Combine(dir, "sorted.safetensors")
        let u8 = torch.zeros ([| 1L |], dtype = torch.ScalarType.Byte, device = torch.CPU)
        let f64 = torch.randn ([| 1L |], dtype = torch.float64, device = torch.CPU)
        let f32 = torch.randn ([| 1L |], dtype = torch.float32, device = torch.CPU)

        SafeTensors.save (Map [ "z_u8", u8; "a_f32", f32; "b_f64", f64 ]) path


        let meta = SafeTensors.loadMeta path

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

        try
            SafeTensors.load path |> ignore
            failwith "Expected exception for invalid header"
        with :? System.InvalidOperationException as ex ->
            ex.Message |> should haveSubstring "start with '{'")
