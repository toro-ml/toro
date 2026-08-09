namespace Toro.Hub

open System
open System.IO
open System.Text
open System.Text.Json
open TorchSharp
open Toro

/// Read and write the SafeTensors binary format.
module SafeTensors =

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

    /// Load all tensors from a .safetensors file.
    let load (filePath: string) : Result<Map<string, Tensor>, ToroError> =
        result {
            let! headerSize, headerJson =
                ToroError.wrap (fun () ->
                    use fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read)
                    use reader = new BinaryReader(fs)
                    let headerSize = reader.ReadUInt64()
                    let headerBytes = reader.ReadBytes(int headerSize)
                    headerSize, Encoding.UTF8.GetString(headerBytes))

            let dataOffset = 8L + int64 headerSize

            use doc = JsonDocument.Parse(headerJson)
            let mutable tensors = Map.empty

            for prop in doc.RootElement.EnumerateObject() do
                if prop.Name <> "__metadata__" then
                    let! dtype = stringToDType (prop.Value.GetProperty("dtype").GetString())
                    let torchDtype = DType.toTorch dtype

                    let shape = [| for s in prop.Value.GetProperty("shape").EnumerateArray() -> s.GetInt64() |]

                    let offsets =
                        prop.Value.GetProperty("data_offsets").EnumerateArray()
                        |> Seq.toArray

                    let startOff = offsets[0].GetInt64()
                    let endOff = offsets[1].GetInt64()
                    let byteLen = int (endOff - startOff)

                    let! t =
                        ToroError.wrap (fun () ->
                            use fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read)

                            fs.Position <- dataOffset + startOff
                            let buf = Array.zeroCreate<byte> byteLen
                            fs.ReadExactly(buf, 0, byteLen)

                            let tensor = torch.zeros (shape, dtype = torchDtype)
                            buf.AsSpan().CopyTo(tensor.bytes)
                            tensor)

                    let! toroTensor = Tensor.ofTorchTensor t
                    tensors <- tensors |> Map.add prop.Name toroTensor

            return tensors
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
                Encoding.UTF8.GetString(ms.ToArray())

            do!
                ToroError.wrap (fun () ->
                    use fs = new FileStream(filePath, FileMode.Create, FileAccess.Write)
                    use bw = new BinaryWriter(fs)
                    let headerBytes = Encoding.UTF8.GetBytes(headerJson)
                    bw.Write(uint64 headerBytes.Length)
                    bw.Write(headerBytes)

                    for inner, _ in entries do
                        let bytes = inner.bytes
                        let buf = Array.zeroCreate<byte> bytes.Length
                        bytes.CopyTo(buf.AsSpan())
                        bw.Write(buf))
        }
