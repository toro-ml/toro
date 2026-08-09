namespace Toro.Text

open Toro

/// Text preprocessing utilities for batching and masking.
module Preprocessing =
    /// Pad or truncate a token list to the specified length.
    let padOrTruncate (maxLen: int) (padId: int) (tokens: int list) : int list =
        let len = tokens.Length

        if len >= maxLen then
            tokens |> List.take maxLen
        else
            tokens @ List.replicate (maxLen - len) padId

    /// Convert a batch of token lists to a tensor, padding each to maxLen.
    let batchToTensor
        (maxLen: int)
        (padId: int)
        (batch: int list list)
        (dtype: DType)
        (device: Device)
        : Result<Tensor, ToroError> =
        let padded =
            batch
            |> List.map (padOrTruncate maxLen padId)
            |> List.map (List.map int64 >> List.toArray)
            |> List.toArray
            |> array2D

        Tensor.ofArray (padded, device)

    /// Generate an attention mask: 1 where token differs from padId, 0 at pad positions.
    let attentionMask (tokens: Tensor) (padId: int) : Tensor =
        let mask = tokens.neScalar (float padId)
        mask
