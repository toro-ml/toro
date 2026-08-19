namespace Toro.Models

open System
open TorchSharp
open Toro
open Toro.Models

/// A fixed-capacity, per-layer DistilGPT-2 key/value cache.
type DistilGpt2Cache internal (config: DistilGpt2Config, batchSize: int64, capacity: int64, dtype, device) =
    do
        DistilGpt2Config.validate config

        if batchSize <= 0L then
            invalidArg (nameof batchSize) "Cache batch size must be positive."

        if capacity <= 0L || capacity > config.MaxPositionEmbeddings then
            invalidArg (nameof capacity) $"Cache capacity must be between 1 and {config.MaxPositionEmbeddings}."

    let headSize = config.EmbeddingSize / config.NumAttentionHeads

    let storage =
        new FixedKvCache(
            nameof DistilGpt2Cache,
            config.NumHiddenLayers,
            batchSize,
            config.NumAttentionHeads,
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

/// Tensor inputs accepted by DistilGPT-2.
type DistilGpt2Input = CausalLmInput<DistilGpt2Cache>

/// Tensor outputs produced by DistilGPT-2.
type DistilGpt2Output = CausalLmOutput<DistilGpt2Cache>
