namespace Toro.Text

open System.Collections.Generic
open System.Collections.ObjectModel

/// Text normalizer configuration.
type TextNormalizer =
    | NoNormalizer
    | LowerCase
    | CustomNormalizer of Microsoft.ML.Tokenizers.Normalizer

/// Text pre-tokenizer configuration.
type TextPreTokenizer =
    | DefaultPreTokenizer
    | ByteLevelPreTokenizer
    | Regex of pattern: string
    | CustomPreTokenizer of Microsoft.ML.Tokenizers.PreTokenizer

/// Configuration for Tiktoken (OpenAI) tokenizers.
type TiktokenConfig = {
    Model: string
    ExtraSpecialTokens: (string * int) list
}

/// Configuration for BPE tokenizers, including optional byte-level encoding.
type BpeConfig = {
    VocabPath: string
    MergesPath: string
    ByteLevel: bool
    SpecialTokens: (string * int) list
    UnknownToken: string option
    ContinuingSubwordPrefix: string option
    EndOfWordSuffix: string option
    FuseUnknownTokens: bool
    PreTokenizer: TextPreTokenizer
    Normalizer: TextNormalizer
}

/// Configuration for WordPiece (BERT) tokenizers.
type WordPieceConfig = {
    VocabPath: string
    SpecialTokens: (string * int) list
    UnknownToken: string
    MaxInputCharsPerWord: int option
    ContinuingSubwordPrefix: string option
    PreTokenizer: TextPreTokenizer
    Normalizer: TextNormalizer
}

/// Configuration for SentencePiece tokenizers.
type SentencePieceConfig = { ModelPath: string }

module TiktokenConfig =
    let create (model: string) : TiktokenConfig = {
        Model = model
        ExtraSpecialTokens = []
    }

module BpeConfig =
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

module WordPieceConfig =
    let create (vocabPath: string) : WordPieceConfig = {
        VocabPath = vocabPath
        SpecialTokens = []
        UnknownToken = "[UNK]"
        MaxInputCharsPerWord = None
        ContinuingSubwordPrefix = Some "##"
        PreTokenizer = DefaultPreTokenizer
        Normalizer = NoNormalizer
    }

module SentencePieceConfig =
    let create (modelPath: string) : SentencePieceConfig = { ModelPath = modelPath }

/// Tokenizer wrapping Microsoft.ML.Tokenizers. Provides F#-idiomatic encode/decode
/// and exposes the underlying instance via .Inner for advanced scenarios.
type Tokenizer internal (inner: Microsoft.ML.Tokenizers.Tokenizer) =

    /// Underlying Microsoft.ML.Tokenizers instance (escape hatch).
    member _.Inner = inner

    /// Encode text to token IDs.
    member _.encode(text: string) : int list =
        inner.EncodeToIds(text, true, true)
        |> Seq.map int
        |> Seq.toList

    /// Decode token IDs to text.
    member _.decode(ids: int list) : string = inner.Decode(ids :> seq<int>)

    /// Count the number of tokens in text without allocating the ID list.
    member _.countTokens(text: string) : int = inner.CountTokens(text, true, true)

module Tokenizer =
    let private emptySpecialTokens =
        ReadOnlyDictionary(dict []) :> IReadOnlyDictionary<string, int>

    let private toSpecialTokensDict (tokens: (string * int) list) =
        if tokens.IsEmpty then
            null
        else
            ReadOnlyDictionary(dict tokens) :> IReadOnlyDictionary<string, int>

    let private toNormalizer =
        function
        | NoNormalizer -> null
        | LowerCase -> Microsoft.ML.Tokenizers.LowerCaseNormalizer() :> Microsoft.ML.Tokenizers.Normalizer
        | CustomNormalizer n -> n

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
        | CustomPreTokenizer p -> p

    /// Create from a Tiktoken config.
    let fromTiktoken (config: TiktokenConfig) : Tokenizer =
        let extra = toSpecialTokensDict config.ExtraSpecialTokens

        let inner =
            Microsoft.ML.Tokenizers.TiktokenTokenizer.CreateForModel(config.Model, extra)

        Tokenizer(inner)

    /// Create from a BPE config.
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

        Tokenizer(inner)

    /// Create from a WordPiece config.
    let fromWordPiece (config: WordPieceConfig) : Tokenizer =
        let opts = Microsoft.ML.Tokenizers.WordPieceOptions()
        let specialTokens = toSpecialTokensDict config.SpecialTokens
        opts.SpecialTokens <- specialTokens
        opts.UnknownToken <- config.UnknownToken

        match config.ContinuingSubwordPrefix with
        | Some p -> opts.ContinuingSubwordPrefix <- p
        | None -> ()

        match config.MaxInputCharsPerWord with
        | Some n -> opts.MaxInputCharsPerWord <- n
        | None -> ()

        let norm = toNormalizer config.Normalizer

        if not (isNull norm) then
            opts.Normalizer <- norm

        let pre = toPreTokenizer specialTokens config.PreTokenizer

        if not (isNull pre) then
            opts.PreTokenizer <- pre

        let inner =
            Microsoft.ML.Tokenizers.WordPieceTokenizer.Create(config.VocabPath, opts)

        Tokenizer(inner)

    /// Create from a SentencePiece config.
    let fromSentencePiece (config: SentencePieceConfig) : Tokenizer =
        use stream = System.IO.File.OpenRead(config.ModelPath)
        let inner = Microsoft.ML.Tokenizers.SentencePieceTokenizer.Create(stream)
        Tokenizer(inner)

    /// Wrap an existing Microsoft.ML.Tokenizers.Tokenizer instance.
    let wrap (inner: Microsoft.ML.Tokenizers.Tokenizer) : Tokenizer = Tokenizer(inner)
