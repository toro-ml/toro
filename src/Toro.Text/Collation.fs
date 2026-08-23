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

/// Target sequence length for padding and truncation.
type CollationLength =
    /// Pad and truncate every sequence to this length.
    | Fixed of int
    /// Pad to the longest sequence in the batch, optionally truncating first.
    | BatchMax of maxLength: int option

/// Text collation policy.
type CollationOptions = {
    Length: CollationLength
    PadTokenId: int64
    PaddingSide: PaddingSide
    TruncationSide: TruncationSide
}

/// Constructors for text collation policies.
module CollationOptions =

    /// Create a right-padded policy that truncates tokens from the right.
    let create padTokenId length = {
        Length = length
        PadTokenId = padTokenId
        PaddingSide = PaddingSide.Right
        TruncationSide = TruncationSide.Right
    }

    let internal validate options =
        match options.Length with
        | Fixed length when length <= 0 -> invalidArg (nameof options) "Collation length must be positive."
        | BatchMax(Some maxLength) when maxLength <= 0 -> invalidArg (nameof options) "Collation max length must be positive."
        | Fixed _
        | BatchMax _ -> ()

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

    let private truncationLimit options =
        match options.Length with
        | Fixed length -> Some length
        | BatchMax maxLength -> maxLength

    let private truncate (options: CollationOptions) (tokens: int64 list) =
        match truncationLimit options with
        | None -> tokens
        | Some limit when tokens.Length <= limit -> tokens
        | Some limit ->
            match options.TruncationSide with
            | TruncationSide.Left -> takeLast limit tokens
            | TruncationSide.Right -> List.take limit tokens

    let private pad (options: CollationOptions) (targetLength: int) (tokens: int64 list) =
        let retainedLength = tokens.Length
        let padding = List.replicate (targetLength - retainedLength) options.PadTokenId

        let padded =
            match options.PaddingSide with
            | PaddingSide.Left -> padding @ tokens
            | PaddingSide.Right -> tokens @ padding

        let attended = List.replicate retainedLength true
        let masked = List.replicate padding.Length false

        let mask =
            match options.PaddingSide with
            | PaddingSide.Left -> masked @ attended
            | PaddingSide.Right -> attended @ masked

        padded, mask

    let private padWidth options (truncated: int64 list list) =
        match options.Length with
        | Fixed length -> length
        | BatchMax _ -> truncated |> List.map _.Length |> List.max

    /// Encode one text into a token tensor. Fixed length pads to that length;
    /// batch-max pads only to the (possibly truncated) sequence itself.
    let toTensor (tokenizer: Tokenizer) (text: string) (options: CollationOptions) (device: torch.Device) : Tensor =
        CollationOptions.validate options
        let truncated = tokenizer.encode text |> truncate options

        let targetLength =
            match options.Length with
            | Fixed length -> length
            | BatchMax _ -> truncated.Length

        let ids, _ = pad options targetLength truncated
        torch.tensor (List.toArray ids, dtype = torch.int64, device = device)

    /// Encode texts into a batch with mask and retained lengths.
    let batch (tokenizer: Tokenizer) (texts: string list) (options: CollationOptions) (device: torch.Device) : EncodedBatch =
        CollationOptions.validate options

        if texts.IsEmpty then
            invalidArg (nameof texts) "At least one text is required for collation."

        let encoded = texts |> List.map tokenizer.encode
        let truncated = encoded |> List.map (truncate options)
        let width = padWidth options truncated
        let padded = truncated |> List.map (pad options width)

        let inputIds =
            padded
            |> List.map (fun (ids, _) -> List.toArray ids)
            |> List.toArray
            |> array2D
            |> fun values -> torch.tensor (values, dtype = torch.int64, device = device)

        let attentionMask =
            padded
            |> List.map (fun (_, mask) -> List.toArray mask)
            |> List.toArray
            |> array2D
            |> fun values -> torch.tensor (values, dtype = torch.bool, device = device)

        {
            InputIds = inputIds
            AttentionMask = attentionMask
            Lengths = encoded |> List.map (_.Length >> int64) |> List.toArray
        }
