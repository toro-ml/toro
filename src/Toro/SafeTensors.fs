namespace Toro

open System
open System.IO
open System.Text
open System.Text.Json
open TorchSharp

/// Metadata for a single tensor entry in a SafeTensors header.
type TensorMeta = {
    DType: torch.ScalarType
    Shape: int64[]
    StartOffset: int64
    EndOffset: int64
}

/// Read and write the SafeTensors binary format.
module SafeTensors =

    let private maxHeaderSize = 100_000_000L

    let private dtypeToString (dtype: torch.ScalarType) : string =
        match dtype with
        | torch.ScalarType.Float16 -> "F16"
        | torch.ScalarType.BFloat16 -> "BF16"
        | torch.ScalarType.Float32 -> "F32"
        | torch.ScalarType.Float64 -> "F64"
        | torch.ScalarType.Int32 -> "I32"
        | torch.ScalarType.Int64 -> "I64"
        | torch.ScalarType.Byte -> "U8"
        | torch.ScalarType.Bool -> "BOOL"
        | other -> raise (NotSupportedException $"Unsupported dtype for SafeTensors: {other}")

    let private stringToDType (s: string) : torch.ScalarType =
        match s with
        | "F16" -> torch.ScalarType.Float16
        | "BF16" -> torch.ScalarType.BFloat16
        | "F32" -> torch.ScalarType.Float32
        | "F64" -> torch.ScalarType.Float64
        | "I32" -> torch.ScalarType.Int32
        | "I64" -> torch.ScalarType.Int64
        | "U8" -> torch.ScalarType.Byte
        | "BOOL" -> torch.ScalarType.Bool
        | other -> raise (NotSupportedException $"Unsupported SafeTensors dtype: {other}")

    let private dtypeByteSize (dtype: torch.ScalarType) : int =
        match dtype with
        | torch.ScalarType.Float16
        | torch.ScalarType.BFloat16 -> 2
        | torch.ScalarType.Float32
        | torch.ScalarType.Int32 -> 4
        | torch.ScalarType.Float64
        | torch.ScalarType.Int64 -> 8
        | torch.ScalarType.Byte
        | torch.ScalarType.Bool -> 1
        | other -> raise (NotSupportedException $"Unsupported dtype for SafeTensors: {other}")

    let private parseEntry (prop: JsonProperty) : string * TensorMeta =
        let dtype = stringToDType (prop.Value.GetProperty("dtype").GetString())

        let shape = [| for s in prop.Value.GetProperty("shape").EnumerateArray() -> s.GetInt64() |]

        if shape |> Array.exists (fun d -> d < 0L) then
            invalidOp $"SafeTensors: tensor '%s{prop.Name}' has negative dimension in shape %A{shape}"

        let offsets =
            prop.Value.GetProperty("data_offsets").EnumerateArray()
            |> Seq.toArray

        let startOff = offsets[0].GetInt64()
        let endOff = offsets[1].GetInt64()

        if endOff < startOff then
            invalidOp $"SafeTensors: tensor '%s{prop.Name}' has invalid offsets: start=%d{startOff} > end=%d{endOff}"

        let nelements = shape |> Array.fold (fun acc d -> d * acc) 1L
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

    let private readTensor (stream: Stream) (dataOffset: int64) (meta: TensorMeta) : torch.Tensor =
        let byteLen = int (meta.EndOffset - meta.StartOffset)

        stream.Position <- dataOffset + meta.StartOffset
        let buf = Array.zeroCreate<byte> byteLen
        stream.ReadExactly(buf, 0, byteLen)
        let t = torch.zeros (meta.Shape, dtype = meta.DType)
        buf.AsSpan().CopyTo(t.bytes)
        t

    /// Load all tensors from a .safetensors file.
    let load (filePath: string) : Map<string, torch.Tensor> =
        use fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read)
        let headerSize, meta = loadMetaFromStream fs
        let dataOffset = 8L + headerSize

        meta |> Map.map (fun _ m -> readTensor fs dataOffset m)

    /// Load only the tensors whose names are in the given set.
    let loadSelected (filePath: string) (names: Set<string>) : Map<string, TensorMeta> * Map<string, torch.Tensor> =
        use fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read)
        let headerSize, meta = loadMetaFromStream fs
        let dataOffset = 8L + headerSize

        let tensors =
            meta
            |> Map.filter (fun key _ -> Set.contains key names)
            |> Map.map (fun _ m -> readTensor fs dataOffset m)

        meta, tensors

    let private saveStaged (tensors: Map<string, torch.Tensor>) (filePath: string) =
        let dir = Path.GetDirectoryName filePath

        if not (String.IsNullOrEmpty dir) && not (Directory.Exists dir) then
            Directory.CreateDirectory dir |> ignore

        let sortedEntries =
            tensors
            |> Map.toArray
            |> Array.sortBy (fun (name, t: torch.Tensor) -> -dtypeByteSize t.dtype, name)

        let entries, _ =
            sortedEntries
            |> Array.mapFold
                (fun offset (name, (tensor: torch.Tensor)) ->
                    // Tensor.bytes is only safe to read from host memory. Stage every
                    // tensor on CPU and normalize its layout before serializing it.
                    let inner = tensor.cpu().contiguous ()
                    let byteLen = inner.NumberOfElements * inner.ElementSize
                    let entry = (name, inner.dtype, inner.shape, offset, offset + byteLen)
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
                    writer.WriteNumberValue(d)

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

    /// Save tensors to a .safetensors file after staging them as contiguous CPU tensors.
    let save (tensors: Map<string, torch.Tensor>) (filePath: string) : unit = scoped { saveStaged tensors filePath }
