namespace Toro

open System
open System.IO
open System.Text
open System.Text.Json
open TorchSharp

/// Metadata for a single tensor entry in a SafeTensors header.
type TensorMeta = {
    DType: DType
    Shape: int list
    StartOffset: int64
    EndOffset: int64
}

/// Read and write the SafeTensors binary format.
module SafeTensors =

    let private maxHeaderSize = 100_000_000L

    let private dtypeToString (dtype: DType) : string =
        match dtype with
        | F16 -> "F16"
        | BF16 -> "BF16"
        | F32 -> "F32"
        | F64 -> "F64"
        | I32 -> "I32"
        | I64 -> "I64"
        | U8 -> "U8"
        | Bool -> "BOOL"

    let private stringToDType (s: string) : DType =
        match s with
        | "F16" -> F16
        | "BF16" -> BF16
        | "F32" -> F32
        | "F64" -> F64
        | "I32" -> I32
        | "I64" -> I64
        | "U8" -> U8
        | "BOOL" -> Bool
        | other -> raise (NotSupportedException $"Unsupported SafeTensors dtype: {other}")

    let private dtypeByteSize (dtype: DType) : int =
        match dtype with
        | F16
        | BF16 -> 2
        | F32
        | I32 -> 4
        | F64
        | I64 -> 8
        | U8
        | Bool -> 1

    let private parseEntry (prop: JsonProperty) : string * TensorMeta =
        let dtype = stringToDType (prop.Value.GetProperty("dtype").GetString())

        let shape = [
            for s in prop.Value.GetProperty("shape").EnumerateArray() -> s.GetInt64() |> int
        ]

        if shape |> List.exists (fun d -> d < 0) then
            invalidOp $"SafeTensors: tensor '%s{prop.Name}' has negative dimension in shape %A{shape}"

        let offsets =
            prop.Value.GetProperty("data_offsets").EnumerateArray()
            |> Seq.toArray

        let startOff = offsets[0].GetInt64()
        let endOff = offsets[1].GetInt64()

        if endOff < startOff then
            invalidOp $"SafeTensors: tensor '%s{prop.Name}' has invalid offsets: start=%d{startOff} > end=%d{endOff}"

        let nelements = shape |> List.fold (fun acc d -> int64 d * acc) 1L
        let expectedBytes = nelements * int64 (dtypeByteSize dtype)
        let actualBytes = endOff - startOff

        if actualBytes <> expectedBytes then
            invalidOp
                $"SafeTensors: tensor '%s{prop.Name}' size mismatch: shape %A{shape} * %s{dtypeToString dtype} = %d{expectedBytes} bytes, but data_offsets span %d{actualBytes} bytes"

        prop.Name,
        {
            DType = dtype
            Shape = shape
            StartOffset = startOff
            EndOffset = endOff
        }

    let private validateOffsetContinuity (entries: (string * TensorMeta) list) : int64 =
        let sorted = entries |> List.sortBy (fun (_, m) -> m.StartOffset)

        sorted
        |> List.fold
            (fun expectedStart (name, m) ->
                if m.StartOffset <> expectedStart then
                    invalidOp
                        $"SafeTensors: tensor '%s{name}' offset gap or overlap: expected start=%d{expectedStart}, got start=%d{m.StartOffset}"

                m.EndOffset)
            0L

    /// Parse the SafeTensors header from an open stream.
    /// Returns (headerSize, tensorMetadata).
    let loadMetaFromStream (stream: Stream) : int64 * Map<string, TensorMeta> =
        let buf = Array.zeroCreate<byte> 8

        if stream.Read(buf, 0, 8) < 8 then
            invalidOp "File too small to contain a SafeTensors header"

        let headerSize = BitConverter.ToUInt64(buf, 0) |> int64

        if headerSize > maxHeaderSize then
            invalidOp $"SafeTensors header too large: %d{headerSize} bytes (max %d{maxHeaderSize})"

        let headerBytes = Array.zeroCreate<byte> (int headerSize)
        stream.ReadExactly(headerBytes, 0, int headerSize)
        let headerJson = Encoding.UTF8.GetString(headerBytes)

        if headerJson.Length > 0 && headerJson[0] <> '{' then
            invalidOp "SafeTensors header must start with '{'"

        use doc = JsonDocument.Parse(headerJson)

        let props =
            doc.RootElement.EnumerateObject()
            |> Seq.filter (fun p -> p.Name <> "__metadata__")
            |> Seq.toList

        let entries = props |> List.map parseEntry

        let lastEndOffset = validateOffsetContinuity entries

        if stream.CanSeek then
            let expectedFileSize = 8L + headerSize + lastEndOffset

            if stream.Length <> expectedFileSize then
                invalidOp $"SafeTensors: file size mismatch: expected %d{expectedFileSize} bytes, got %d{stream.Length} bytes"

        let meta = entries |> Map.ofList
        headerSize, meta

    /// Load only the header metadata from a .safetensors file.
    let loadMeta (filePath: string) : Map<string, TensorMeta> =
        use fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read)
        let _, meta = loadMetaFromStream fs
        meta

    let private readTensor (stream: Stream) (dataOffset: int64) (meta: TensorMeta) : Tensor =
        let torchDtype = DType.toTorch meta.DType
        let shape = meta.Shape |> List.map int64 |> Array.ofList
        let byteLen = int (meta.EndOffset - meta.StartOffset)

        stream.Position <- dataOffset + meta.StartOffset
        let buf = Array.zeroCreate<byte> byteLen
        stream.ReadExactly(buf, 0, byteLen)
        let t = torch.zeros (shape, dtype = torchDtype)
        buf.AsSpan().CopyTo(t.bytes)
        Tensor.ofTorchTensor t

    /// Load all tensors from a .safetensors file.
    let load (filePath: string) : Map<string, Tensor> =
        use fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read)
        let headerSize, meta = loadMetaFromStream fs
        let dataOffset = 8L + headerSize

        meta |> Map.map (fun _ m -> readTensor fs dataOffset m)

    /// Load only the tensors whose names are in the given set.
    let loadSelected (filePath: string) (names: Set<string>) : Map<string, TensorMeta> * Map<string, Tensor> =
        use fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read)
        let headerSize, meta = loadMetaFromStream fs
        let dataOffset = 8L + headerSize

        let tensors =
            meta
            |> Map.filter (fun key _ -> Set.contains key names)
            |> Map.map (fun _ m -> readTensor fs dataOffset m)

        meta, tensors

    /// Save tensors to a .safetensors file.
    let save (tensors: Map<string, Tensor>) (filePath: string) : unit =
        let dir = Path.GetDirectoryName filePath

        if not (String.IsNullOrEmpty dir) && not (Directory.Exists dir) then
            Directory.CreateDirectory dir |> ignore

        let sortedEntries =
            tensors
            |> Map.toArray
            |> Array.sortBy (fun (name, t: Tensor) -> -dtypeByteSize t.DType, name)

        let entries, _ =
            sortedEntries
            |> Array.mapFold
                (fun offset (name, (tensor: Tensor)) ->
                    let inner = tensor.Inner.contiguous ()
                    let byteLen = inner.NumberOfElements * inner.ElementSize
                    let entry = (name, tensor.DType, tensor.Shape, offset, offset + byteLen)
                    (inner, entry), offset + byteLen)
                0L

        let headerJson =
            use ms = new MemoryStream()
            use writer = new Utf8JsonWriter(ms)
            writer.WriteStartObject()

            for _, (name, dtype, shape, startOff, endOff) in entries do
                writer.WriteStartObject(name)
                writer.WriteString("dtype", dtypeToString dtype)
                writer.WriteStartArray("shape")

                for d in shape do
                    writer.WriteNumberValue(int64 d)

                writer.WriteEndArray()
                writer.WriteStartArray("data_offsets")
                writer.WriteNumberValue(startOff)
                writer.WriteNumberValue(endOff)
                writer.WriteEndArray()
                writer.WriteEndObject()

            writer.WriteEndObject()
            writer.Flush()
            ms.ToArray()

        let paddedLen = (headerJson.Length + 7) / 8 * 8
        let padded = Array.zeroCreate<byte> paddedLen
        Array.Copy(headerJson, padded, headerJson.Length)

        for i in headerJson.Length .. paddedLen - 1 do
            padded[i] <- 0x20uy

        use fs = new FileStream(filePath, FileMode.Create, FileAccess.Write)
        use bw = new BinaryWriter(fs)
        bw.Write(uint64 paddedLen)
        bw.Write(padded)

        for inner, _ in entries do
            let bytes = inner.bytes
            let buf = Array.zeroCreate<byte> bytes.Length
            bytes.CopyTo(buf.AsSpan())
            bw.Write(buf)
