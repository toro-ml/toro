namespace Toro

open System
open System.Collections.Generic
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

type internal SafeTensorLocation = {
    Stream: FileStream
    DataOffset: int64
    Meta: TensorMeta
}

/// Validated, single-threaded reader for one SafeTensors file or a sharded
/// SafeTensors index. Tensors returned by ReadTensor are owned by the caller.
type SafeTensorReader
    internal (metadata: Map<string, TensorMeta>, locations: Map<string, SafeTensorLocation>, streams: FileStream list) =
    let mutable disposed = false

    /// Metadata for every tensor exposed by this reader.
    member _.Metadata = metadata

    /// Read one tensor into a newly allocated CPU Tensor.
    member _.ReadTensor(name: string) : Tensor =
        if disposed then
            raise (ObjectDisposedException(nameof SafeTensorReader))

        let location =
            match Map.tryFind name locations with
            | Some value -> value
            | None -> raise (KeyNotFoundException $"SafeTensors tensor '{name}' was not found.")

        location.Stream.Position <- location.DataOffset + location.Meta.StartOffset

        let tensor =
            torch.empty (location.Meta.Shape, dtype = location.Meta.DType, device = torch.CPU)

        try
            location.Stream.ReadExactly(tensor.bytes)
            tensor
        with _ ->
            tensor.Dispose()
            reraise ()

    interface IDisposable with
        member _.Dispose() =
            if not disposed then
                disposed <- true
                streams |> List.iter _.Dispose()

/// Read and write the SafeTensors binary format.
module SafeTensors =

    let private maxHeaderSize = 100_000_000UL

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

    let private stringToDType (value: string) : torch.ScalarType =
        match value with
        | "F16" -> torch.ScalarType.Float16
        | "BF16" -> torch.ScalarType.BFloat16
        | "F32" -> torch.ScalarType.Float32
        | "F64" -> torch.ScalarType.Float64
        | "I32" -> torch.ScalarType.Int32
        | "I64" -> torch.ScalarType.Int64
        | "U8" -> torch.ScalarType.Byte
        | "BOOL" -> torch.ScalarType.Bool
        | other -> raise (NotSupportedException $"Unsupported SafeTensors dtype: {other}")

    let private dtypeByteSize (dtype: torch.ScalarType) : int64 =
        match dtype with
        | torch.ScalarType.Float16
        | torch.ScalarType.BFloat16 -> 2L
        | torch.ScalarType.Float32
        | torch.ScalarType.Int32 -> 4L
        | torch.ScalarType.Float64
        | torch.ScalarType.Int64 -> 8L
        | torch.ScalarType.Byte
        | torch.ScalarType.Bool -> 1L
        | other -> raise (NotSupportedException $"Unsupported dtype for SafeTensors: {other}")

    let private multiplyChecked description left right =
        if left <> 0L && right > Int64.MaxValue / left then
            invalidOp $"SafeTensors: {description} exceeds Int64 capacity."

        left * right

    let private parseEntry (property: JsonProperty) : string * TensorMeta =
        if property.Value.ValueKind <> JsonValueKind.Object then
            invalidOp $"SafeTensors: tensor '%s{property.Name}' metadata must be an object."

        let dtype = stringToDType (property.Value.GetProperty("dtype").GetString())

        let shape = [|
            for dimension in property.Value.GetProperty("shape").EnumerateArray() -> dimension.GetInt64()
        |]

        if shape |> Array.exists (fun dimension -> dimension < 0L) then
            invalidOp $"SafeTensors: tensor '%s{property.Name}' has negative dimension in shape %A{shape}."

        let offsets =
            property.Value.GetProperty("data_offsets").EnumerateArray()
            |> Seq.map _.GetInt64()
            |> Seq.toArray

        if offsets.Length <> 2 then
            invalidOp $"SafeTensors: tensor '%s{property.Name}' data_offsets must contain exactly two values."

        let startOffset = offsets[0]
        let endOffset = offsets[1]

        if startOffset < 0L || endOffset < startOffset then
            invalidOp $"SafeTensors: tensor '%s{property.Name}' has invalid offsets [%d{startOffset}, %d{endOffset}]."

        let elementCount =
            shape
            |> Array.fold (multiplyChecked $"tensor '{property.Name}' element count") 1L

        let expectedBytes =
            multiplyChecked $"tensor '{property.Name}' byte count" elementCount (dtypeByteSize dtype)

        let actualBytes = endOffset - startOffset

        if actualBytes <> expectedBytes then
            invalidOp
                $"SafeTensors: tensor '%s{property.Name}' size mismatch: shape %A{shape} * %s{dtypeToString dtype} = %d{expectedBytes} bytes, but data_offsets span %d{actualBytes} bytes."

        property.Name,
        {
            DType = dtype
            Shape = shape
            StartOffset = startOffset
            EndOffset = endOffset
        }

    let private validateOffsetContinuity (entries: (string * TensorMeta) list) : int64 =
        entries
        |> List.sortBy (fun (_, metadata) -> metadata.StartOffset, metadata.EndOffset)
        |> List.fold
            (fun expectedStart (name, metadata) ->
                if metadata.StartOffset <> expectedStart then
                    invalidOp
                        $"SafeTensors: tensor '%s{name}' offset gap or overlap: expected start=%d{expectedStart}, got start=%d{metadata.StartOffset}."

                metadata.EndOffset)
            0L

    let private validateUniqueProperties context (properties: JsonProperty list) =
        let names = HashSet<string>(StringComparer.Ordinal)

        for property in properties do
            if not (names.Add property.Name) then
                invalidOp $"{context} contains duplicate key '{property.Name}'."

    /// Parse the SafeTensors header from an open stream.
    /// Returns (headerSize, tensorMetadata).
    let loadMetaFromStream (stream: Stream) : int64 * Map<string, TensorMeta> =
        let sizeBytes = Array.zeroCreate<byte> 8
        stream.ReadExactly(sizeBytes)
        let headerSizeValue = BitConverter.ToUInt64(sizeBytes, 0)

        if headerSizeValue > maxHeaderSize then
            invalidOp $"SafeTensors header too large: {headerSizeValue} bytes (max {maxHeaderSize})."

        let headerSize = int headerSizeValue
        let headerBytes = Array.zeroCreate<byte> headerSize
        stream.ReadExactly(headerBytes)
        let headerJson = Encoding.UTF8.GetString(headerBytes)

        if String.IsNullOrEmpty headerJson || headerJson[0] <> '{' then
            invalidOp "SafeTensors header must start with '{'."

        use document = JsonDocument.Parse(headerJson)

        if document.RootElement.ValueKind <> JsonValueKind.Object then
            invalidOp "SafeTensors header must be a JSON object."

        let properties = document.RootElement.EnumerateObject() |> Seq.toList
        validateUniqueProperties "SafeTensors header" properties

        properties
        |> List.tryFind (fun property -> property.Name = "__metadata__")
        |> Option.iter (fun property ->
            if property.Value.ValueKind <> JsonValueKind.Object then
                invalidOp "SafeTensors __metadata__ must be an object."

            let metadataProperties = property.Value.EnumerateObject() |> Seq.toList
            validateUniqueProperties "SafeTensors __metadata__" metadataProperties

            if
                metadataProperties
                |> List.exists (fun item -> item.Value.ValueKind <> JsonValueKind.String)
            then
                invalidOp "SafeTensors __metadata__ values must be strings.")

        let entries =
            properties
            |> List.filter (fun property -> property.Name <> "__metadata__")
            |> List.map parseEntry

        let lastEndOffset = validateOffsetContinuity entries

        if stream.CanSeek then
            let dataOffset = 8L + int64 headerSize

            if lastEndOffset > Int64.MaxValue - dataOffset then
                invalidOp "SafeTensors file size exceeds Int64 capacity."

            let expectedFileSize = dataOffset + lastEndOffset

            if stream.Length <> expectedFileSize then
                invalidOp $"SafeTensors: file size mismatch: expected %d{expectedFileSize} bytes, got %d{stream.Length} bytes."

        int64 headerSize, entries |> Map.ofList

    let private openValidatedFile filePath =
        let stream =
            new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read)

        try
            let headerSize, metadata = loadMetaFromStream stream
            stream, 8L + headerSize, metadata
        with _ ->
            stream.Dispose()
            reraise ()

    let private validateShardName (indexPath: string) (filename: string) =
        if String.IsNullOrWhiteSpace filename then
            invalidOp $"SafeTensors index '{indexPath}' contains an empty shard filename."

        if filename.Contains('\\') || Path.IsPathRooted filename then
            invalidOp $"SafeTensors index '{indexPath}' contains unsafe shard filename '{filename}'."

        let segments = filename.Split('/', StringSplitOptions.None)

        if
            segments
            |> Array.exists (fun segment ->
                String.IsNullOrEmpty segment
                || segment = "."
                || segment = ".."
                || segment.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0)
        then
            invalidOp $"SafeTensors index '{indexPath}' contains unsafe shard filename '{filename}'."

        filename

    let private parseIndex indexPath =
        use document = JsonDocument.Parse(File.ReadAllText indexPath)

        if document.RootElement.ValueKind <> JsonValueKind.Object then
            invalidOp $"SafeTensors index '{indexPath}' must be a JSON object."

        let rootProperties = document.RootElement.EnumerateObject() |> Seq.toList
        validateUniqueProperties $"SafeTensors index '{indexPath}'" rootProperties

        let weightMap = document.RootElement.GetProperty("weight_map")

        if weightMap.ValueKind <> JsonValueKind.Object then
            invalidOp $"SafeTensors index '{indexPath}' weight_map must be an object."

        let mappings = weightMap.EnumerateObject() |> Seq.toList
        validateUniqueProperties $"SafeTensors index '{indexPath}' weight_map" mappings

        mappings
        |> List.map (fun property ->
            if String.IsNullOrWhiteSpace property.Name then
                invalidOp $"SafeTensors index '{indexPath}' contains an empty tensor name."

            if property.Value.ValueKind <> JsonValueKind.String then
                invalidOp $"SafeTensors index '{indexPath}' mapping for '{property.Name}' must be a string."

            property.Name, validateShardName indexPath (property.Value.GetString()))
        |> Map.ofList

    let private shardPath (indexPath: string) (filename: string) =
        let indexDirectory = indexPath |> Path.GetFullPath |> Path.GetDirectoryName

        let path =
            filename.Split('/')
            |> Array.append [| indexDirectory |]
            |> Path.Combine
            |> Path.GetFullPath

        let prefix =
            indexDirectory.TrimEnd(Path.DirectorySeparatorChar)
            + string Path.DirectorySeparatorChar

        if not (path.StartsWith(prefix, StringComparison.Ordinal)) then
            invalidOp $"SafeTensors index '{indexPath}' resolves shard '{filename}' outside its directory."

        path

    /// Return the distinct shard filenames referenced by an index, sorted ordinally.
    let indexShardFiles (indexPath: string) : string list =
        parseIndex indexPath
        |> Map.values
        |> Seq.distinct
        |> Seq.sortWith (fun left right -> StringComparer.Ordinal.Compare(left, right))
        |> Seq.toList

    /// Open and validate one SafeTensors file for name-based tensor reads.
    let openFile (filePath: string) : SafeTensorReader =
        let stream, dataOffset, metadata = openValidatedFile filePath

        let locations =
            metadata
            |> Map.map (fun _ tensorMetadata -> {
                Stream = stream
                DataOffset = dataOffset
                Meta = tensorMetadata
            })

        new SafeTensorReader(metadata, locations, [ stream ])

    /// Open and validate a sharded SafeTensors index and every referenced shard.
    let openIndex (indexPath: string) : SafeTensorReader =
        let weightMap = parseIndex indexPath
        let opened = ResizeArray<FileStream>()

        try
            let shards =
                weightMap
                |> Map.values
                |> Seq.distinct
                |> Seq.map (fun filename ->
                    let stream, dataOffset, metadata = openValidatedFile (shardPath indexPath filename)
                    opened.Add stream
                    filename, (stream, dataOffset, metadata))
                |> Map.ofSeq

            for KeyValue(filename, (_, _, metadata)) in shards do
                let expected =
                    weightMap
                    |> Map.filter (fun _ mappedFilename -> mappedFilename = filename)
                    |> Map.keys
                    |> Set.ofSeq

                let actual = metadata |> Map.keys |> Set.ofSeq

                if expected <> actual then
                    let missing = Set.difference expected actual |> Set.toList
                    let unexpected = Set.difference actual expected |> Set.toList

                    invalidOp
                        $"SafeTensors shard '{filename}' does not match its index: missing=%A{missing}; unexpected=%A{unexpected}."

            let locations =
                weightMap
                |> Map.map (fun name filename ->
                    let stream, dataOffset, metadata = shards[filename]

                    match Map.tryFind name metadata with
                    | Some tensorMetadata -> {
                        Stream = stream
                        DataOffset = dataOffset
                        Meta = tensorMetadata
                      }
                    | None -> invalidOp $"SafeTensors shard '{filename}' does not contain indexed tensor '{name}'.")

            let metadata = locations |> Map.map (fun _ location -> location.Meta)
            new SafeTensorReader(metadata, locations, opened |> Seq.toList)
        with _ ->
            opened |> Seq.iter _.Dispose()
            reraise ()

    /// Load only the header metadata from a .safetensors file.
    let loadMeta (filePath: string) : Map<string, TensorMeta> =
        use reader = openFile filePath
        reader.Metadata

    /// Load all tensors from a .safetensors file.
    let load (filePath: string) : Map<string, Tensor> =
        use reader = openFile filePath

        scoped {
            return
                reader.Metadata
                |> Map.map (fun name _ -> reader.ReadTensor name)
        }

    /// Load only the tensors whose names are in the given set.
    let loadSelected (filePath: string) (names: Set<string>) : Map<string, TensorMeta> * Map<string, Tensor> =
        use reader = openFile filePath

        scoped {
            let tensors =
                reader.Metadata
                |> Map.filter (fun name _ -> Set.contains name names)
                |> Map.map (fun name _ -> reader.ReadTensor name)

            return reader.Metadata, tensors
        }

    let private saveStaged (tensors: Map<string, Tensor>) (filePath: string) =
        let directory = Path.GetDirectoryName filePath

        if
            not (String.IsNullOrEmpty directory)
            && not (Directory.Exists directory)
        then
            Directory.CreateDirectory directory |> ignore

        let sortedEntries =
            tensors
            |> Map.toArray
            |> Array.sortBy (fun (name, tensor: Tensor) -> -dtypeByteSize tensor.dtype, name)

        let entries, _ =
            sortedEntries
            |> Array.mapFold
                (fun offset (name, tensor: Tensor) ->
                    let inner = tensor.cpu().contiguous ()
                    let byteLength = inner.NumberOfElements * inner.ElementSize
                    let entry = name, inner.dtype, inner.shape, offset, offset + byteLength
                    (inner, entry), offset + byteLength)
                0L

        let headerJson =
            use stream = new MemoryStream()
            use writer = new Utf8JsonWriter(stream)
            writer.WriteStartObject()

            for _, (name, dtype, shape, startOffset, endOffset) in entries do
                writer.WriteStartObject(name)
                writer.WriteString("dtype", dtypeToString dtype)
                writer.WriteStartArray("shape")

                for dimension in shape do
                    writer.WriteNumberValue(dimension)

                writer.WriteEndArray()
                writer.WriteStartArray("data_offsets")
                writer.WriteNumberValue(startOffset)
                writer.WriteNumberValue(endOffset)
                writer.WriteEndArray()
                writer.WriteEndObject()

            writer.WriteEndObject()
            writer.Flush()
            stream.ToArray()

        let paddedLength = (headerJson.Length + 7) / 8 * 8
        let padded = Array.zeroCreate<byte> paddedLength
        Array.Copy(headerJson, padded, headerJson.Length)

        for index in headerJson.Length .. paddedLength - 1 do
            padded[index] <- 0x20uy

        use stream = new FileStream(filePath, FileMode.Create, FileAccess.Write)
        use writer = new BinaryWriter(stream)
        writer.Write(uint64 paddedLength)
        writer.Write(padded)

        for inner, _ in entries do
            let bytes = inner.bytes
            let buffer = Array.zeroCreate<byte> bytes.Length
            bytes.CopyTo(buffer.AsSpan())
            writer.Write(buffer)

    /// Save tensors to a .safetensors file after staging them as contiguous CPU tensors.
    let save (tensors: Map<string, Tensor>) (filePath: string) : unit = scoped { saveStaged tensors filePath }
