module TextTests

open Xunit
open FsUnit.Xunit
open Toro
open Toro.Text
open TestHelper

// --- Tokenizer.wrap and .Inner ---

[<Fact>]
let ``Tokenizer wrap exposes Inner`` () =
    let vocab = [| "hello"; "world"; "[UNK]" |]

    let vocabStream =
        new System.IO.MemoryStream(System.Text.Encoding.UTF8.GetBytes(vocab |> String.concat "\n"))

    let specialTokens =
        System.Collections.ObjectModel.ReadOnlyDictionary(dict [ "[UNK]", 2 ])
        :> System.Collections.Generic.IReadOnlyDictionary<string, int>

    let opts =
        Microsoft.ML.Tokenizers.WordPieceOptions(SpecialTokens = specialTokens, UnknownToken = "[UNK]")

    let mlTok = Microsoft.ML.Tokenizers.WordPieceTokenizer.Create(vocabStream, opts)
    let tok = Tokenizer.wrap mlTok
    tok.Inner |> should not' (be null)

[<Fact>]
let ``Tokenizer encode and decode round-trip`` () =
    let vocab = [| "hello"; "world"; "[UNK]" |]

    let vocabStream =
        new System.IO.MemoryStream(System.Text.Encoding.UTF8.GetBytes(vocab |> String.concat "\n"))

    let specialTokens =
        System.Collections.ObjectModel.ReadOnlyDictionary(dict [ "[UNK]", 2 ])
        :> System.Collections.Generic.IReadOnlyDictionary<string, int>

    let opts =
        Microsoft.ML.Tokenizers.WordPieceOptions(SpecialTokens = specialTokens, UnknownToken = "[UNK]")

    let mlTok = Microsoft.ML.Tokenizers.WordPieceTokenizer.Create(vocabStream, opts)
    let tok = Tokenizer.wrap mlTok
    let ids = tok.encode "hello world"
    ids |> should equal [ 0; 1 ]
    let decoded = tok.decode ids
    decoded |> should equal "hello world"

[<Fact>]
let ``Tokenizer countTokens returns correct count`` () =
    let vocab = [| "hello"; "world"; "[UNK]" |]

    let vocabStream =
        new System.IO.MemoryStream(System.Text.Encoding.UTF8.GetBytes(vocab |> String.concat "\n"))

    let specialTokens =
        System.Collections.ObjectModel.ReadOnlyDictionary(dict [ "[UNK]", 2 ])
        :> System.Collections.Generic.IReadOnlyDictionary<string, int>

    let opts =
        Microsoft.ML.Tokenizers.WordPieceOptions(SpecialTokens = specialTokens, UnknownToken = "[UNK]")

    let mlTok = Microsoft.ML.Tokenizers.WordPieceTokenizer.Create(vocabStream, opts)
    let tok = Tokenizer.wrap mlTok
    tok.countTokens "hello world" |> should equal 2

// --- Config-based factories ---

let private writeVocabFile (vocab: string array) =
    let path = System.IO.Path.GetTempFileName()
    System.IO.File.WriteAllLines(path, vocab)
    path

[<Fact>]
let ``fromWordPiece with config creates working tokenizer`` () =
    let path = writeVocabFile [| "hello"; "world"; "[UNK]" |]

    try
        let tok =
            Tokenizer.fromWordPiece {
                WordPieceConfig.create path with
                    SpecialTokens = [ "[UNK]", 2 ]
            }

        let ids = tok.encode "hello world"
        ids |> should equal [ 0; 1 ]
    finally
        System.IO.File.Delete(path)

[<Fact>]
let ``fromWordPiece with LowerCase normalizer`` () =
    let path = writeVocabFile [| "hello"; "world"; "[UNK]" |]

    try
        let tok =
            Tokenizer.fromWordPiece {
                WordPieceConfig.create path with
                    SpecialTokens = [ "[UNK]", 2 ]
                    Normalizer = LowerCase
            }

        let ids = tok.encode "HELLO WORLD"
        ids |> should equal [ 0; 1 ]
    finally
        System.IO.File.Delete(path)

[<Fact>]
let ``fromWordPiece with Regex pre-tokenizer`` () =
    let path = writeVocabFile [| "hello"; "world"; "!"; "[UNK]" |]

    try
        let tok =
            Tokenizer.fromWordPiece {
                WordPieceConfig.create path with
                    SpecialTokens = [ "[UNK]", 3 ]
                    PreTokenizer = Regex @"\w+|[^\w\s]+"
            }

        let ids = tok.encode "hello!"
        ids |> should equal [ 0; 2 ]
    finally
        System.IO.File.Delete(path)

// --- Encode module ---

[<Fact>]
let ``Encode.toTensor produces correct shape`` () =
    let path = writeVocabFile [| "a"; "b"; "c"; "d"; "[UNK]" |]

    try
        let tok =
            Tokenizer.fromWordPiece {
                WordPieceConfig.create path with
                    SpecialTokens = [ "[UNK]", 4 ]
            }

        let t = Encode.toTensor tok "a b" 5 0 Cpu |> unwrap
        t.Shape |> should equal [ 5 ]
    finally
        System.IO.File.Delete(path)

[<Fact>]
let ``Encode.batch produces correct shapes`` () =
    let path = writeVocabFile [| "a"; "b"; "c"; "d"; "[UNK]" |]

    try
        let tok =
            Tokenizer.fromWordPiece {
                WordPieceConfig.create path with
                    SpecialTokens = [ "[UNK]", 4 ]
            }

        let struct (ids, mask) = Encode.batch tok [ "a b"; "c" ] 4 0 Cpu |> unwrap
        ids.Shape |> should equal [ 2; 4 ]
        mask.Shape |> should equal [ 2; 4 ]
    finally
        System.IO.File.Delete(path)

[<Fact>]
let ``Encode.attentionMask marks pad positions as zero`` () =
    let data: int64 array = [| 1L; 2L; 0L; 0L |]
    let t = Tensor.ofArray (data, Cpu) |> unwrap

    let mask = Encode.attentionMask t 0
    let m0 = mask.at [ I 0 ] |> scalarF32
    let m1 = mask.at [ I 1 ] |> scalarF32
    let m2 = mask.at [ I 2 ] |> scalarF32
    let m3 = mask.at [ I 3 ] |> scalarF32
    m0 |> should (equalWithin 1e-5) 1.0f
    m1 |> should (equalWithin 1e-5) 1.0f
    m2 |> should (equalWithin 1e-5) 0.0f
    m3 |> should (equalWithin 1e-5) 0.0f
