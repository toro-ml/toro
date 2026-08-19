namespace Toro.Text

open TorchSharp
open Toro

/// Side on which padding tokens are inserted.
type PaddingSide =
    | Left
    | Right

/// Side removed when a token sequence exceeds the target length.
type TruncationSide =
    | Left
    | Right

/// Fixed-length text collation policy.
type CollationOptions = {
    Length: int
    PadTokenId: int64
    PaddingSide: PaddingSide
    TruncationSide: TruncationSide
}

/// Constructors for text collation policies.
module CollationOptions =

    /// Create a right-padded policy that truncates tokens from the right.
    let create length padTokenId = {
        Length = length
        PadTokenId = padTokenId
        PaddingSide = PaddingSide.Right
        TruncationSide = TruncationSide.Right
    }

    let internal validate options =
        if options.Length <= 0 then
            invalidArg (nameof options) "Collation length must be positive."

/// A model-ready batch produced from text.
type EncodedBatch = {
    /// Token IDs with shape [batch, sequence].
    InputIds: Tensor
    /// Boolean attention mask with shape [batch, sequence].
    AttentionMask: Tensor
    /// Original token count for each batch item before truncation.
    Lengths: int64 array
}

/// Text-to-tensor collation.
module Collation =

    let private takeLast (length: int) (values: 'Value list) =
        values |> List.skip (values.Length - length)

    let private resize (options: CollationOptions) (tokens: int64 list) =
        let retained =
            if tokens.Length <= options.Length then
                tokens
            else
                match options.TruncationSide with
                | TruncationSide.Left -> takeLast options.Length tokens
                | TruncationSide.Right -> List.take options.Length tokens

        let retainedLength = retained.Length
        let padding = List.replicate (options.Length - retainedLength) options.PadTokenId

        let padded =
            match options.PaddingSide with
            | PaddingSide.Left -> padding @ retained
            | PaddingSide.Right -> retained @ padding

        let attended = List.replicate retainedLength true
        let masked = List.replicate padding.Length false

        let mask =
            match options.PaddingSide with
            | PaddingSide.Left -> masked @ attended
            | PaddingSide.Right -> attended @ masked

        padded, mask, int64 tokens.Length

    /// Encode one text into a fixed-length token tensor.
    let toTensor (tokenizer: Tokenizer) (text: string) (options: CollationOptions) (device: torch.Device) : Tensor =
        CollationOptions.validate options
        let ids, _, _ = tokenizer.encode text |> resize options
        torch.tensor (List.toArray ids, dtype = torch.int64, device = device)

    /// Encode texts into a fixed-length batch with mask and retained lengths.
    let batch (tokenizer: Tokenizer) (texts: string list) (options: CollationOptions) (device: torch.Device) : EncodedBatch =
        CollationOptions.validate options

        if texts.IsEmpty then
            invalidArg (nameof texts) "At least one text is required for collation."

        let rows = texts |> List.map (tokenizer.encode >> resize options)

        let inputIds =
            rows
            |> List.map (fun (ids, _, _) -> List.toArray ids)
            |> List.toArray
            |> array2D
            |> fun values -> torch.tensor (values, dtype = torch.int64, device = device)

        let attentionMask =
            rows
            |> List.map (fun (_, mask, _) -> List.toArray mask)
            |> List.toArray
            |> array2D
            |> fun values -> torch.tensor (values, dtype = torch.bool, device = device)

        {
            InputIds = inputIds
            AttentionMask = attentionMask
            Lengths =
                rows
                |> List.map (fun (_, _, length) -> length)
                |> List.toArray
        }
