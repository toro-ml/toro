namespace Toro.Models

open System
open TorchSharp
open Toro
open Toro.Models
open Toro.Models.Interop

/// A fixed-capacity, per-layer SmolLM2 key/value cache.
type SmolLm2Cache internal (config: SmolLm2Config, batchSize: int64, capacity: int64, dtype, device) =
    do
        SmolLm2Config.validate config

        if batchSize <= 0L then
            invalidArg (nameof batchSize) "Cache batch size must be positive."

        if capacity <= 0L || capacity > config.MaxPositionEmbeddings then
            invalidArg (nameof capacity) $"Cache capacity must be between 1 and {config.MaxPositionEmbeddings}."

    let headSize = config.HiddenSize / config.NumAttentionHeads

    let storage =
        new FixedKvCache(
            nameof SmolLm2Cache,
            config.NumHiddenLayers,
            batchSize,
            config.NumKeyValueHeads,
            capacity,
            headSize,
            dtype,
            device
        )

    /// Number of batch items stored by this cache.
    member _.BatchSize = storage.BatchSize

    /// Maximum number of tokens stored by this cache.
    member _.Capacity = storage.Capacity

    /// Number of tokens currently stored by this cache.
    member _.Length = storage.Length

    /// Remove all logical entries without reallocating storage.
    member _.Reset() = storage.Reset()

    member internal _.Validate(batch: int64, sequenceLength: int64) = storage.Validate(batch, sequenceLength)

    member internal _.Append(layerIndex: int, start: int64, key: Tensor, value: Tensor) =
        storage.Append(layerIndex, start, key, value)

    member internal _.Advance(sequenceLength: int64) = storage.Advance sequenceLength

    interface IDisposable with
        member _.Dispose() = (storage :> IDisposable).Dispose()

/// Tensor inputs accepted by SmolLM2.
type SmolLm2Input = CausalLmInput<SmolLm2Cache>

/// Tensor outputs produced by SmolLM2.
type SmolLm2Output = CausalLmOutput<SmolLm2Cache>
