module TextTests

open Xunit
open FsUnit.Xunit
open Toro
open TorchSharp
open Toro.Text
open TestHelper

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
    tok |> should not' (be null)

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
    ids |> should equal [ 0L; 1L ]
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

[<Fact>]
let ``incremental decoder preserves WordPiece spacing`` () =
    let vocab = [| "hello"; "world"; "[UNK]" |]

    let vocabStream =
        new System.IO.MemoryStream(System.Text.Encoding.UTF8.GetBytes(vocab |> String.concat "\n"))

    let options = Microsoft.ML.Tokenizers.WordPieceOptions(UnknownToken = "[UNK]")

    let tokenizer =
        Microsoft.ML.Tokenizers.WordPieceTokenizer.Create(vocabStream, options)
        |> Tokenizer.wrap

    let decoder = tokenizer.createDecoder ()

    decoder.append 0L |> should equal "hello"
    decoder.append 1L |> should equal " world"
    decoder.complete () |> should equal ""

[<Fact>]
let ``incremental ByteLevel decoder waits for complete UTF-8`` () =
    let vocabPath = System.IO.Path.GetTempFileName()
    let mergesPath = System.IO.Path.GetTempFileName()

    try
        System.IO.File.WriteAllText(vocabPath, """{"ã":0,"ģ":1,"Ĥ":2}""")
        System.IO.File.WriteAllText(mergesPath, "#version: 0.2\n")

        let tokenizer =
            Tokenizer.fromBpe {
                BpeConfig.create vocabPath mergesPath with
                    ByteLevel = true
            }

        let ids = tokenizer.encode "あ"
        ids |> should equal [ 0L; 1L; 2L ]
        let decoder = tokenizer.createDecoder ()
        decoder.append ids[0] |> should equal ""
        decoder.append ids[1] |> should equal ""
        decoder.append ids[2] |> should equal "あ"
        decoder.complete () |> should equal ""
    finally
        System.IO.File.Delete(vocabPath)
        System.IO.File.Delete(mergesPath)

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
        ids |> should equal [ 0L; 1L ]
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
        ids |> should equal [ 0L; 1L ]
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
        ids |> should equal [ 0L; 2L ]
    finally
        System.IO.File.Delete(path)

// --- Collation module ---

[<Fact>]
let ``Collation.toTensor produces correct shape`` () =
    let path = writeVocabFile [| "a"; "b"; "c"; "d"; "[UNK]" |]

    try
        let tok =
            Tokenizer.fromWordPiece {
                WordPieceConfig.create path with
                    SpecialTokens = [ "[UNK]", 4 ]
            }

        let options = CollationOptions.create 5 0L
        let t = Collation.toTensor tok "a b" options torch.CPU
        t.shape |> should equal [| 5L |]
    finally
        System.IO.File.Delete(path)

[<Fact>]
let ``Collation.batch produces correct shapes and lengths`` () =
    let path = writeVocabFile [| "a"; "b"; "c"; "d"; "[UNK]" |]

    try
        let tok =
            Tokenizer.fromWordPiece {
                WordPieceConfig.create path with
                    SpecialTokens = [ "[UNK]", 4 ]
            }

        let options = CollationOptions.create 4 0L
        let batch = Collation.batch tok [ "a b"; "c" ] options torch.CPU
        batch.InputIds.shape |> should equal [| 2L; 4L |]
        batch.AttentionMask.shape |> should equal [| 2L; 4L |]
        batch.Lengths |> should equal [| 2L; 1L |]
    finally
        System.IO.File.Delete(path)

[<Fact>]
let ``Collation supports left padding and left truncation`` () =
    let path = writeVocabFile [| "a"; "b"; "c"; "d"; "[UNK]" |]

    try
        let tokenizer =
            Tokenizer.fromWordPiece {
                WordPieceConfig.create path with
                    SpecialTokens = [ "[UNK]", 4L ]
            }

        let options = {
            CollationOptions.create 3 9L with
                PaddingSide = PaddingSide.Left
                TruncationSide = TruncationSide.Left
        }

        let short = Collation.batch tokenizer [ "a" ] options torch.CPU
        let shortIds = short.InputIds.flatten().data<int64>().ToArray()
        let shortMask = short.AttentionMask.flatten().data<bool>().ToArray()
        shortIds |> should equal [| 9L; 9L; 0L |]
        shortMask |> should equal [| false; false; true |]

        let long = Collation.batch tokenizer [ "a b c d" ] options torch.CPU

        long.InputIds.flatten().data<int64>().ToArray()
        |> should equal [| 1L; 2L; 3L |]

        long.Lengths |> should equal [| 4L |]
    finally
        System.IO.File.Delete(path)

[<Fact>]
let ``Collation rejects invalid length and empty batch`` () =
    let path = writeVocabFile [| "a"; "[UNK]" |]

    try
        let tokenizer =
            Tokenizer.fromWordPiece {
                WordPieceConfig.create path with
                    SpecialTokens = [ "[UNK]", 1L ]
            }

        let invalid = CollationOptions.create 0 0L
        shouldFail (fun () -> Collation.toTensor tokenizer "a" invalid torch.CPU |> ignore)

        let valid = CollationOptions.create 2 0L
        shouldFail (fun () -> Collation.batch tokenizer [] valid torch.CPU |> ignore)
    finally
        System.IO.File.Delete(path)
