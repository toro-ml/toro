namespace Toro.Text

open System
open System.Buffers
open System.Collections.Generic
open System.Collections.ObjectModel

/// Request-local state for decoding token IDs into text fragments.
type IncrementalDecoder internal (append: int64 -> string, complete: unit -> string) =
    let mutable completed = false

    /// Decode one additional token ID into zero or more complete characters.
    member _.append(tokenId: int64) =
        if completed then
            invalidOp "The incremental decoder has already completed."

        append tokenId

    /// Flush any remaining text. Subsequent calls return an empty string.
    member _.complete() =
        if completed then
            ""
        else
            completed <- true
            complete ()

/// Text tokenizer with token IDs represented as 64-bit integers.
type Tokenizer
    internal
    (
        encode: string -> int64 list,
        decode: int64 list -> string,
        countTokens: string -> int,
        backend: obj,
        createDecoder: unit -> IncrementalDecoder
    ) =

    /// Encode text to token IDs.
    member _.encode(text: string) : int64 list = encode text

    /// Decode token IDs to text.
    member _.decode(ids: int64 list) : string = decode ids

    /// Count the number of tokens in text.
    member _.countTokens(text: string) : int = countTokens text

    /// Create request-local state for incremental token decoding.
    member _.createDecoder() = createDecoder ()

    member internal _.Backend = backend

/// Text normalizer configuration for Microsoft.ML.Tokenizers.
type TextNormalizer =
    | NoNormalizer
    | LowerCase
    | CustomNormalizer of Microsoft.ML.Tokenizers.Normalizer

/// Text pre-tokenizer configuration for Microsoft.ML.Tokenizers.
type TextPreTokenizer =
    | DefaultPreTokenizer
    | ByteLevelPreTokenizer
    | Regex of pattern: string
    | CustomPreTokenizer of Microsoft.ML.Tokenizers.PreTokenizer

/// Configuration for a Microsoft.ML.Tokenizers Tiktoken tokenizer.
type TiktokenConfig = {
    Model: string
    ExtraSpecialTokens: (string * int64) list
}

/// Configuration for a Microsoft.ML.Tokenizers BPE tokenizer.
type BpeConfig = {
    VocabPath: string
    MergesPath: string
    ByteLevel: bool
    SpecialTokens: (string * int64) list
    UnknownToken: string option
    ContinuingSubwordPrefix: string option
    EndOfWordSuffix: string option
    FuseUnknownTokens: bool
    PreTokenizer: TextPreTokenizer
    Normalizer: TextNormalizer
}

/// Configuration for a Microsoft.ML.Tokenizers WordPiece tokenizer.
type WordPieceConfig = {
    VocabPath: string
    SpecialTokens: (string * int64) list
    UnknownToken: string
    MaxInputCharsPerWord: int option
    ContinuingSubwordPrefix: string option
    PreTokenizer: TextPreTokenizer
    Normalizer: TextNormalizer
}

/// Configuration for a Microsoft.ML.Tokenizers SentencePiece tokenizer.
type SentencePieceConfig = { ModelPath: string }

/// Constructors for Tiktoken configuration.
module TiktokenConfig =

    /// Create a Tiktoken configuration.
    let create (model: string) : TiktokenConfig = {
        Model = model
        ExtraSpecialTokens = []
    }

/// Constructors for BPE configuration.
module BpeConfig =

    /// Create a BPE configuration.
    let create (vocabPath: string) (mergesPath: string) : BpeConfig = {
        VocabPath = vocabPath
        MergesPath = mergesPath
        ByteLevel = false
        SpecialTokens = []
        UnknownToken = None
        ContinuingSubwordPrefix = None
        EndOfWordSuffix = None
        FuseUnknownTokens = false
        PreTokenizer = DefaultPreTokenizer
        Normalizer = NoNormalizer
    }

/// Constructors for WordPiece configuration.
module WordPieceConfig =

    /// Create a WordPiece configuration.
    let create (vocabPath: string) : WordPieceConfig = {
        VocabPath = vocabPath
        SpecialTokens = []
        UnknownToken = "[UNK]"
        MaxInputCharsPerWord = None
        ContinuingSubwordPrefix = Some "##"
        PreTokenizer = DefaultPreTokenizer
        Normalizer = NoNormalizer
    }

/// Constructors for SentencePiece configuration.
module SentencePieceConfig =

    /// Create a SentencePiece configuration.
    let create (modelPath: string) : SentencePieceConfig = { ModelPath = modelPath }

/// Microsoft.ML.Tokenizers adapters and factories.
module Tokenizer =

    let private emptySpecialTokens =
        ReadOnlyDictionary(dict []) :> IReadOnlyDictionary<string, int>

    let private backendId parameterName tokenId =
        if
            tokenId < int64 Int32.MinValue
            || tokenId > int64 Int32.MaxValue
        then
            invalidArg parameterName $"Token ID {tokenId} is outside the Microsoft.ML.Tokenizers Int32 range."

        int tokenId

    let private toSpecialTokensDict (tokens: (string * int64) list) =
        if tokens.IsEmpty then
            null
        else
            tokens
            |> Seq.map (fun (token, tokenId) -> token, backendId (nameof tokens) tokenId)
            |> dict
            |> ReadOnlyDictionary
            :> IReadOnlyDictionary<string, int>

    let private toNormalizer =
        function
        | NoNormalizer -> null
        | LowerCase -> Microsoft.ML.Tokenizers.LowerCaseNormalizer() :> Microsoft.ML.Tokenizers.Normalizer
        | CustomNormalizer normalizer -> normalizer

    let private toPreTokenizer specialTokens =
        let specialTokens =
            if isNull specialTokens then
                emptySpecialTokens
            else
                specialTokens

        function
        | DefaultPreTokenizer -> null
        | ByteLevelPreTokenizer ->
            let pattern =
                "'(?:[sdmt]|ll|ve|re)| ?(?>\\p{L}+)| ?(?>\\p{N}+)| ?(?>[^\\s\\p{L}\\p{N}]+)|(?>\\s+)$|\\s+(?!\\S)|\\s"

            Microsoft.ML.Tokenizers.RegexPreTokenizer(System.Text.RegularExpressions.Regex(pattern), specialTokens)
            :> Microsoft.ML.Tokenizers.PreTokenizer
        | Regex pattern ->
            Microsoft.ML.Tokenizers.RegexPreTokenizer(System.Text.RegularExpressions.Regex(pattern), specialTokens)
            :> Microsoft.ML.Tokenizers.PreTokenizer
        | CustomPreTokenizer preTokenizer -> preTokenizer

    let rec private stablePrefixLength minimumLength (decoded: string) length =
        if length > minimumLength && decoded[length - 1] = '\uFFFD' then
            stablePrefixLength minimumLength decoded (length - 1)
        else
            length

    let private differentialDecoder (decode: int64 list -> string) =
        let tokenIds = ResizeArray<int64>()
        let mutable emitted = ""

        let next finished =
            let decoded = tokenIds |> Seq.toList |> decode

            if not (decoded.StartsWith(emitted, StringComparison.Ordinal)) then
                invalidOp "The token decoder changed text that was already emitted."

            let nextLength =
                if finished then
                    decoded.Length
                else
                    stablePrefixLength emitted.Length decoded decoded.Length

            let delta = decoded.Substring(emitted.Length, nextLength - emitted.Length)
            emitted <- decoded.Substring(0, nextLength)
            delta

        IncrementalDecoder(
            (fun tokenId ->
                tokenIds.Add tokenId
                next false),
            fun () -> next true
        )

    let private spanDecoder (inner: Microsoft.ML.Tokenizers.Tokenizer) =
        let pending = ResizeArray<int>()

        let decodePending () =
            let mutable buffer = Array.zeroCreate<char> (max 128 (pending.Count * 8))
            let mutable decoded = None

            while decoded.IsNone do
                let mutable idsConsumed = 0
                let mutable charsWritten = 0

                match inner.Decode(pending, buffer.AsSpan(), &idsConsumed, &charsWritten) with
                | OperationStatus.Done ->
                    pending.RemoveRange(0, idsConsumed)
                    decoded <- Some(String(buffer, 0, charsWritten))
                | OperationStatus.DestinationTooSmall -> buffer <- Array.zeroCreate (buffer.Length * 2)
                | status -> invalidOp $"Incremental token decoding failed with status {status}."

            decoded.Value

        let append tokenId =
            pending.Add(backendId (nameof tokenId) tokenId)
            decodePending ()

        let complete () =
            if pending.Count = 0 then
                ""
            else
                let decoded = inner.Decode pending
                pending.Clear()
                decoded

        IncrementalDecoder(append, complete)

    let private wrapWith decoderFactory (inner: Microsoft.ML.Tokenizers.Tokenizer) : Tokenizer =
        if isNull inner then
            nullArg (nameof inner)

        let encode (text: string) =
            inner.EncodeToIds(text, true, true)
            |> Seq.map int64
            |> Seq.toList

        let decode (ids: int64 list) =
            ids |> Seq.map (backendId (nameof ids)) |> inner.Decode

        Tokenizer(
            encode,
            decode,
            (fun (text: string) -> inner.CountTokens(text, true, true)),
            inner,
            (fun () -> decoderFactory decode)
        )

    /// Wrap a Microsoft.ML.Tokenizers tokenizer.
    let wrap (inner: Microsoft.ML.Tokenizers.Tokenizer) : Tokenizer = wrapWith differentialDecoder inner

    /// Return the wrapped Microsoft.ML.Tokenizers instance.
    let inner (tokenizer: Tokenizer) : Microsoft.ML.Tokenizers.Tokenizer =
        tokenizer.Backend :?> Microsoft.ML.Tokenizers.Tokenizer

    /// Create a Tiktoken tokenizer.
    let fromTiktoken (config: TiktokenConfig) : Tokenizer =
        let extra = toSpecialTokensDict config.ExtraSpecialTokens

        let inner =
            Microsoft.ML.Tokenizers.TiktokenTokenizer.CreateForModel(config.Model, extra)

        inner |> wrapWith (fun _ -> spanDecoder inner)

    /// Create a BPE tokenizer.
    let fromBpe (config: BpeConfig) : Tokenizer =
        let options =
            Microsoft.ML.Tokenizers.BpeOptions(config.VocabPath, config.MergesPath)

        let specialTokens = toSpecialTokensDict config.SpecialTokens
        options.ByteLevel <- config.ByteLevel
        options.PreTokenizer <- toPreTokenizer specialTokens config.PreTokenizer
        options.Normalizer <- toNormalizer config.Normalizer
        options.SpecialTokens <- specialTokens
        options.UnknownToken <- config.UnknownToken |> Option.toObj
        options.ContinuingSubwordPrefix <- config.ContinuingSubwordPrefix |> Option.toObj
        options.EndOfWordSuffix <- config.EndOfWordSuffix |> Option.toObj
        options.FuseUnknownTokens <- config.FuseUnknownTokens
        let inner = Microsoft.ML.Tokenizers.BpeTokenizer.Create(options)

        if config.ByteLevel then
            inner |> wrapWith (fun _ -> spanDecoder inner)
        else
            wrap inner

    /// Create a WordPiece tokenizer.
    let fromWordPiece (config: WordPieceConfig) : Tokenizer =
        let options = Microsoft.ML.Tokenizers.WordPieceOptions()
        let specialTokens = toSpecialTokensDict config.SpecialTokens
        options.SpecialTokens <- specialTokens
        options.UnknownToken <- config.UnknownToken

        config.ContinuingSubwordPrefix
        |> Option.iter (fun prefix -> options.ContinuingSubwordPrefix <- prefix)

        config.MaxInputCharsPerWord
        |> Option.iter (fun maxChars -> options.MaxInputCharsPerWord <- maxChars)

        toNormalizer config.Normalizer
        |> Option.ofObj
        |> Option.iter (fun normalizer -> options.Normalizer <- normalizer)

        toPreTokenizer specialTokens config.PreTokenizer
        |> Option.ofObj
        |> Option.iter (fun preTokenizer -> options.PreTokenizer <- preTokenizer)

        Microsoft.ML.Tokenizers.WordPieceTokenizer.Create(config.VocabPath, options)
        |> wrap

    /// Create a SentencePiece tokenizer.
    let fromSentencePiece (config: SentencePieceConfig) : Tokenizer =
        use stream = System.IO.File.OpenRead(config.ModelPath)

        Microsoft.ML.Tokenizers.SentencePieceTokenizer.Create(stream)
        |> wrap
