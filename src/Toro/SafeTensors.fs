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

    let private stringToDType (s: string) : Result<DType, ToroError> =
        match s with
        | "F16" -> Ok F16
        | "BF16" -> Ok BF16
        | "F32" -> Ok F32
        | "F64" -> Ok F64
        | "I32" -> Ok I32
        | "I64" -> Ok I64
        | "U8" -> Ok U8
        | "BOOL" -> Ok Bool
        | other -> Error(UnsupportedDType other)

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

    /// Parse the SafeTensors header from an open stream.
    /// Returns (headerSize, tensorMetadata).
    let loadMetaFromStream (stream: Stream) : Result<int64 * Map<string, TensorMeta>, ToroError> =
        result {
            let! headerSize =
                ToroError.wrap (fun () ->
                    let buf = Array.zeroCreate<byte> 8

                    if stream.Read(buf, 0, 8) < 8 then
                        failwith "File too small to contain a SafeTensors header"

                    BitConverter.ToUInt64(buf, 0) |> int64)

            if headerSize > maxHeaderSize then
                return! Error(Msg $"SafeTensors header too large: %d{headerSize} bytes (max %d{maxHeaderSize})")

            let! headerJson =
                ToroError.wrap (fun () ->
                    let headerBytes = Array.zeroCreate<byte> (int headerSize)
                    stream.ReadExactly(headerBytes, 0, int headerSize)
                    Encoding.UTF8.GetString(headerBytes))

            if headerJson.Length > 0 && headerJson[0] <> '{' then
                return! Error(Msg "SafeTensors header must start with '{'")

            use doc = JsonDocument.Parse(headerJson)
            let mutable meta = Map.empty

            for prop in doc.RootElement.EnumerateObject() do
                if prop.Name <> "__metadata__" then
                    let! dtype = stringToDType (prop.Value.GetProperty("dtype").GetString())

                    let shape = [
                        for s in prop.Value.GetProperty("shape").EnumerateArray() -> s.GetInt64() |> int
                    ]

                    let offsets =
                        prop.Value.GetProperty("data_offsets").EnumerateArray()
                        |> Seq.toArray

                    let startOff = offsets[0].GetInt64()
                    let endOff = offsets[1].GetInt64()

                    if endOff < startOff then
                        return!
                            Error(
                                Msg
                                    $"SafeTensors: tensor '%s{prop.Name}' has invalid offsets: start=%d{startOff} > end=%d{endOff}"
                            )

                    let expectedBytes =
                        int64 (shape |> List.fold (fun acc d -> acc * (max d 1)) 1)
                        * int64 (dtypeByteSize dtype)

                    let actualBytes = endOff - startOff

                    if actualBytes <> expectedBytes then
                        return!
                            Error(
                                Msg
                                    $"SafeTensors: tensor '%s{prop.Name}' size mismatch: shape %A{shape} * %s{dtypeToString dtype} = %d{expectedBytes} bytes, but data_offsets span %d{actualBytes} bytes"
                            )

                    meta <-
                        meta
                        |> Map.add prop.Name {
                            DType = dtype
                            Shape = shape
                            StartOffset = startOff
                            EndOffset = endOff
                        }

            return headerSize, meta
        }

    /// Load only the header metadata from a .safetensors file.
    let loadMeta (filePath: string) : Result<Map<string, TensorMeta>, ToroError> =
        result {
            use fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read)
            let! _, meta = loadMetaFromStream fs
            return meta
        }

    let private readTensor (stream: Stream) (dataOffset: int64) (meta: TensorMeta) : Result<Tensor, ToroError> =
        result {
            let torchDtype = DType.toTorch meta.DType
            let shape = meta.Shape |> List.map int64 |> Array.ofList
            let byteLen = int (meta.EndOffset - meta.StartOffset)

            let! t =
                ToroError.wrap (fun () ->
                    stream.Position <- dataOffset + meta.StartOffset
                    let buf = Array.zeroCreate<byte> byteLen
                    stream.ReadExactly(buf, 0, byteLen)
                    let tensor = torch.zeros (shape, dtype = torchDtype)
                    buf.AsSpan().CopyTo(tensor.bytes)
                    tensor)

            return! Tensor.ofTorchTensor t
        }

    /// Load all tensors from a .safetensors file.
    let load (filePath: string) : Result<Map<string, Tensor>, ToroError> =
        result {
            use fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read)
            let! headerSize, meta = loadMetaFromStream fs
            let dataOffset = 8L + headerSize
            let mutable tensors = Map.empty

            for kv in meta do
                let! t = readTensor fs dataOffset kv.Value
                tensors <- tensors |> Map.add kv.Key t

            return tensors
        }

    /// Load only the tensors whose names are in the given set.
    let loadSelected
        (filePath: string)
        (names: Set<string>)
        : Result<Map<string, TensorMeta> * Map<string, Tensor>, ToroError> =
        result {
            use fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read)
            let! headerSize, meta = loadMetaFromStream fs
            let dataOffset = 8L + headerSize
            let mutable tensors = Map.empty

            for kv in meta do
                if Set.contains kv.Key names then
                    let! t = readTensor fs dataOffset kv.Value
                    tensors <- tensors |> Map.add kv.Key t

            return meta, tensors
        }

    /// Save tensors to a .safetensors file.
    let save (tensors: Map<string, Tensor>) (filePath: string) : Result<unit, ToroError> =
        result {
            let dir = Path.GetDirectoryName filePath

            if not (String.IsNullOrEmpty dir) && not (Directory.Exists dir) then
                Directory.CreateDirectory dir |> ignore

            let sortedEntries = tensors |> Map.toArray |> Array.sortBy fst

            let mutable offset = 0L

            let entries =
                sortedEntries
                |> Array.map (fun (name, (tensor: Tensor)) ->
                    let inner = tensor.Inner.contiguous ()
                    let byteLen = inner.NumberOfElements * inner.ElementSize
                    let entry = (name, tensor.DType, tensor.Shape, offset, offset + byteLen)
                    offset <- offset + byteLen
                    (inner, entry))

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

            do!
                ToroError.wrap (fun () ->
                    use fs = new FileStream(filePath, FileMode.Create, FileAccess.Write)
                    use bw = new BinaryWriter(fs)
                    bw.Write(uint64 paddedLen)
                    bw.Write(padded)

                    for inner, _ in entries do
                        let bytes = inner.bytes
                        let buf = Array.zeroCreate<byte> bytes.Length
                        bytes.CopyTo(buf.AsSpan())
                        bw.Write(buf))
        }
