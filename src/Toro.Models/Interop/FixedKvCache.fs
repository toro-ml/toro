namespace Toro.Models.Interop

open System
open System.ComponentModel
open TorchSharp
open Toro

/// Fixed-capacity key/value storage shared by Toro causal model-family packages.
[<Sealed; EditorBrowsable(EditorBrowsableState.Never)>]
type FixedKvCache
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

    /// Number of batch items stored by the cache.
    member _.BatchSize = batchSize

    /// Maximum number of tokens stored by the cache.
    member _.Capacity = capacity

    /// Number of tokens currently stored by the cache.
    member _.Length = length

    /// Remove all logical entries without reallocating storage.
    member _.Reset() =
        ensureAvailable ()
        length <- 0L

    /// Validate an input batch and sequence against the cache state.
    member _.Validate(batch: int64, sequenceLength: int64) =
        ensureAvailable ()

        if batch <> batchSize then
            invalidArg (nameof batch) $"Cache batch size is {batchSize}, but input batch size is {batch}."

        if length + sequenceLength > capacity then
            invalidOp $"KV cache capacity {capacity} is too small for {length + sequenceLength} tokens."

    /// Append one layer's key and value tensors and return the populated views.
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

    /// Advance the logical cache length after every layer has appended.
    member _.Advance(sequenceLength: int64) =
        ensureAvailable ()
        length <- length + sequenceLength

    interface IDisposable with
        member _.Dispose() =
            if not disposed then
                disposed <- true
                keys |> Array.iter _.Dispose()
                values |> Array.iter _.Dispose()
