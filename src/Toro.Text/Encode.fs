namespace Toro.Text

open Toro

/// Text-to-tensor encoding utilities.
module Encode =
    /// Pad or truncate a token list to the specified length.
    let private padOrTruncate (maxLen: int) (padId: int) (tokens: int list) : int list =
        let len = tokens.Length

        if len >= maxLen then
            tokens |> List.take maxLen
        else
            tokens @ List.replicate (maxLen - len) padId

    /// Generate an attention mask: 1 where token differs from padId, 0 at pad positions.
    let attentionMask (tokens: Tensor) (padId: int) : Tensor = tokens.neScalar (float padId)

    /// Encode a single text to a 1-D token ID tensor of length maxLen.
    let toTensor (tokenizer: Tokenizer) (text: string) (maxLen: int) (padId: int) (device: Device) : Result<Tensor, ToroError> =
        let ids = tokenizer.encode text |> padOrTruncate maxLen padId
        let data = ids |> List.map int64 |> List.toArray
        Tensor.ofArray (data, device)

    /// Encode a batch of texts to a padded [B, L] tensor and an attention mask.
    let batch
        (tokenizer: Tokenizer)
        (texts: string list)
        (maxLen: int)
        (padId: int)
        (device: Device)
        : Result<struct (Tensor * Tensor), ToroError> =
        let encoded =
            texts
            |> List.map (fun t -> tokenizer.encode t |> padOrTruncate maxLen padId)

        let data =
            encoded
            |> List.map (List.map int64 >> List.toArray)
            |> List.toArray
            |> array2D

        result {
            let! ids = Tensor.ofArray (data, device)
            let mask = attentionMask ids padId
            return struct (ids, mask)
        }
