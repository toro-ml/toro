namespace Toro.Models

open System
open System.Collections.Generic
open System.IO
open System.Text.Json
open TorchSharp
open Toro
open Toro.NN

module internal JsonConfig =

    let validateObject (label: string) (root: JsonElement) =
        if root.ValueKind <> JsonValueKind.Object then
            invalidOp $"{label} config must be a JSON object."

        root.EnumerateObject()
        |> Seq.countBy _.Name
        |> Seq.tryFind (fun (_, count) -> count > 1)
        |> Option.iter (fun (name, _) -> invalidOp $"{label} config contains duplicate key '{name}'.")

    let tryProperty (root: JsonElement) (name: string) =
        match root.TryGetProperty name with
        | true, value -> Some value
        | false, _ -> None

    let property (label: string) (root: JsonElement) (name: string) =
        tryProperty root name
        |> Option.defaultWith (fun () -> invalidOp $"{label} config is missing '{name}'.")

    let int64Element (label: string) (name: string) (value: JsonElement) =
        match value.TryGetInt64() with
        | true, result -> result
        | false, _ -> invalidOp $"{label} config '{name}' must be an integer."

    let int64Value label (root: JsonElement) name =
        property label root name |> int64Element label name

    let floatValue label (root: JsonElement) name =
        let value = property label root name

        if value.ValueKind <> JsonValueKind.Number then
            invalidOp $"{label} config '{name}' must be a number."

        value.GetDouble()

    let boolValue label (root: JsonElement) name =
        match (property label root name).ValueKind with
        | JsonValueKind.True -> true
        | JsonValueKind.False -> false
        | _ -> invalidOp $"{label} config '{name}' must be a boolean."

    let stringValue label (root: JsonElement) name =
        let value = property label root name

        if value.ValueKind <> JsonValueKind.String then
            invalidOp $"{label} config '{name}' must be a string."

        value.GetString()

module internal LocalModelAssets =

    let openReader (label: string) (directory: string) =
        if String.IsNullOrWhiteSpace directory then
            invalidArg (nameof directory) "Model directory must not be empty."

        let directory = Path.GetFullPath directory

        if not (Directory.Exists directory) then
            invalidArg (nameof directory) $"Model directory does not exist: '{directory}'."

        let configPath = Path.Combine(directory, "config.json")

        if not (File.Exists configPath) then
            invalidOp $"{label} config is missing: '{configPath}'."

        let singlePath = Path.Combine(directory, "model.safetensors")
        let indexPath = Path.Combine(directory, "model.safetensors.index.json")

        let reader =
            match File.Exists singlePath, File.Exists indexPath with
            | true, false -> SafeTensors.openFile singlePath
            | false, true -> SafeTensors.openIndex indexPath
            | true, true -> invalidOp "Model directory contains both single-file and sharded SafeTensors state."
            | false, false -> invalidOp "Model directory contains neither model.safetensors nor model.safetensors.index.json."

        configPath, reader

    let dtype (label: string) (tensorName: string) (reader: SafeTensorReader) =
        reader.Metadata
        |> Map.tryFind tensorName
        |> Option.map _.DType
        |> Option.defaultWith (fun () -> invalidOp $"{label} state is missing '{tensorName}'.")

module internal TensorOwner =

    let disposeDistinct (namedTensors: 'Owner -> seq<string * Tensor>) (owner: 'Owner) =
        let seen = HashSet<obj>(ReferenceEqualityComparer.Instance)

        for _, tensor in namedTensors owner do
            if seen.Add(box tensor) then
                tensor.Dispose()

type internal FixedKvCache
    (ownerName: string, layerCount: int, batchSize: int64, headCount: int64, capacity: int64, headSize: int64, dtype, device) =

    let keys =
        Array.init layerCount (fun _ ->
            torch.empty ([| batchSize; headCount; capacity; headSize |], dtype = dtype, device = device))

    let values =
        Array.init layerCount (fun _ ->
            torch.empty ([| batchSize; headCount; capacity; headSize |], dtype = dtype, device = device))

    let mutable length = 0L
    let mutable disposed = false

    let ensureAvailable () =
        if disposed then
            raise (ObjectDisposedException ownerName)

    member _.BatchSize = batchSize
    member _.Capacity = capacity
    member _.Length = length

    member _.Reset() =
        ensureAvailable ()
        length <- 0L

    member _.Validate(batch: int64, sequenceLength: int64) =
        ensureAvailable ()

        if batch <> batchSize then
            invalidArg (nameof batch) $"Cache batch size is {batchSize}, but input batch size is {batch}."

        if length + sequenceLength > capacity then
            invalidOp $"KV cache capacity {capacity} is too small for {length + sequenceLength} tokens."

    member _.Append(layerIndex: int, start: int64, key: Tensor, value: Tensor) =
        ensureAvailable ()

        if start <> length then
            invalidOp $"KV cache append started at {start}, but the current length is {length}."

        let sequenceLength = key.shape[2]
        use keyDestination = keys[layerIndex].narrow (2L, start, sequenceLength)
        use valueDestination = values[layerIndex].narrow (2L, start, sequenceLength)
        keyDestination.copyInPlace key
        valueDestination.copyInPlace value

        keys[layerIndex].narrow (2L, 0L, start + sequenceLength), values[layerIndex].narrow (2L, 0L, start + sequenceLength)

    member _.Advance(sequenceLength: int64) =
        ensureAvailable ()
        length <- length + sequenceLength

    interface IDisposable with
        member _.Dispose() =
            if not disposed then
                disposed <- true
                keys |> Array.iter _.Dispose()
                values |> Array.iter _.Dispose()

type internal PreparedCausalInput = {
    SequenceLength: int64
    CacheStart: int64
    PositionIds: Tensor
    AttentionMask: Tensor option
    IsCausal: bool
}

module internal CausalInput =

    let private attentionMask (input: CausalLmInput<'Cache>) batchSize sequenceLength start totalLength =
        let paddingMask =
            input.AttentionMask
            |> Option.map (fun mask ->
                if mask.shape <> [| batchSize; totalLength |] then
                    invalidArg (nameof input.AttentionMask) $"Attention mask shape must be [{batchSize}, {totalLength}]."

                mask.to_type(torch.ScalarType.Bool).unsqueeze(1L).unsqueeze (1L))

        let needsExplicitCausalMask = start > 0L && sequenceLength > 1L

        match paddingMask, needsExplicitCausalMask with
        | None, false -> None, start = 0L
        | Some padding, false when sequenceLength = 1L -> Some padding, false
        | padding, _ ->
            let keyPositions =
                torch.arange (totalLength, dtype = torch.int64, device = input.InputIds.device)

            let queryPositions =
                torch.arange (start, start + sequenceLength, dtype = torch.int64, device = input.InputIds.device)

            let causal =
                (keyPositions.unsqueeze (0L)
                 .<=. queryPositions.unsqueeze (1L))
                    .unsqueeze(0L)
                    .unsqueeze (0L)

            match padding with
            | Some value -> Some(causal.logical_and value), false
            | None -> Some causal, false

    let prepare
        (modelName: string)
        (maxPositions: int64)
        (cacheLength: 'Cache -> int64)
        (validateCache: 'Cache -> int64 -> int64 -> unit)
        (input: CausalLmInput<'Cache>)
        : PreparedCausalInput =
        if input.InputIds.dtype <> torch.int64 then
            invalidArg (nameof input.InputIds) $"{modelName} input IDs must use int64 dtype."

        if input.InputIds.shape.Length <> 2 then
            invalidArg (nameof input.InputIds) $"{modelName} input IDs must have shape [batch, sequence]."

        let batchSize = input.InputIds.shape[0]
        let sequenceLength = input.InputIds.shape[1]

        if batchSize <= 0L || sequenceLength <= 0L then
            invalidArg (nameof input.InputIds) $"{modelName} input dimensions must be positive."

        let cacheStart =
            input.Cache
            |> Option.map cacheLength
            |> Option.defaultValue 0L

        input.Cache
        |> Option.iter (fun cache -> validateCache cache batchSize sequenceLength)

        if cacheStart + sequenceLength > maxPositions then
            invalidArg (nameof input.InputIds) $"{modelName} accepts at most {maxPositions} positions."

        let positionIds =
            match input.PositionIds with
            | None ->
                torch.arange (cacheStart, cacheStart + sequenceLength, dtype = torch.int64, device = input.InputIds.device)
            | Some positions ->
                if positions.dtype <> torch.int64 then
                    invalidArg (nameof input.PositionIds) $"{modelName} position IDs must use int64 dtype."

                if
                    positions.shape <> [| sequenceLength |]
                    && positions.shape <> [| batchSize; sequenceLength |]
                then
                    invalidArg
                        (nameof input.PositionIds)
                        $"{modelName} position IDs must have shape [sequence] or [batch, sequence]."

                positions

        let totalLength = cacheStart + sequenceLength

        let mask, isCausal =
            attentionMask input batchSize sequenceLength cacheStart totalLength

        {
            SequenceLength = sequenceLength
            CacheStart = cacheStart
            PositionIds = positionIds
            AttentionMask = mask
            IsCausal = isCausal
        }
