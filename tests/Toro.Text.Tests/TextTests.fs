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

[<Fact>]
let ``incremental ByteLevel decoder complete withholds incomplete UTF-8`` () =
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

        tokenizer.createDecoder().appendAll (ids |> List.truncate 2)
        |> Seq.toList
        |> should be Empty
    finally
        System.IO.File.Delete(vocabPath)
        System.IO.File.Delete(mergesPath)

[<Fact>]
let ``incremental Tiktoken decoder waits for complete UTF-8`` () =
    let tokenizer = Tokenizer.fromTiktoken (TiktokenConfig.create "gpt-4")
    let ids = tokenizer.encode "⭐"
    let prefixes = ids |> List.truncate (ids.Length - 1)
    ids.Length |> should be (greaterThan 1)

    let decoder = tokenizer.createDecoder ()

    prefixes
    |> List.map decoder.append
    |> should equal (List.replicate prefixes.Length "")

    decoder.append (List.last ids) |> should equal "⭐"
    decoder.complete () |> should equal ""

[<Fact>]
let ``incremental Tiktoken decoder complete withholds incomplete UTF-8`` () =
    let tokenizer = Tokenizer.fromTiktoken (TiktokenConfig.create "gpt-4")
    let ids = tokenizer.encode "⭐"
    ids.Length |> should be (greaterThan 1)

    tokenizer.createDecoder().appendAll (ids |> List.truncate (ids.Length - 1))
    |> Seq.toList
    |> should be Empty

[<Fact>]
let ``incremental decoder appendAll yields complete fragments`` () =
    let vocab = [| "hello"; "world"; "[UNK]" |]

    let vocabStream =
        new System.IO.MemoryStream(System.Text.Encoding.UTF8.GetBytes(vocab |> String.concat "\n"))

    let options = Microsoft.ML.Tokenizers.WordPieceOptions(UnknownToken = "[UNK]")

    let tokenizer =
        Microsoft.ML.Tokenizers.WordPieceTokenizer.Create(vocabStream, options)
        |> Tokenizer.wrap

    tokenizer.createDecoder().appendAll [ 0L; 1L ]
    |> Seq.toList
    |> should equal [ "hello"; " world" ]

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

        let options = CollationOptions.create 0L (Fixed 5)
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

        let options = CollationOptions.create 0L (Fixed 4)
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
            CollationOptions.create 9L (Fixed 3) with
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

        let invalid = CollationOptions.create 0L (Fixed 0)
        shouldFail (fun () -> Collation.toTensor tokenizer "a" invalid torch.CPU |> ignore)

        let valid = CollationOptions.create 0L (Fixed 2)
        shouldFail (fun () -> Collation.batch tokenizer [] valid torch.CPU |> ignore)

        let invalidMax = CollationOptions.create 0L (BatchMax(Some 0))

        shouldFail (fun () ->
            Collation.toTensor tokenizer "a" invalidMax torch.CPU
            |> ignore)
    finally
        System.IO.File.Delete(path)

[<Fact>]
let ``Collation.batch pads to the longest sequence when using BatchMax`` () =
    let path = writeVocabFile [| "a"; "b"; "c"; "d"; "[UNK]" |]

    try
        let tok =
            Tokenizer.fromWordPiece {
                WordPieceConfig.create path with
                    SpecialTokens = [ "[UNK]", 4 ]
            }

        let options = CollationOptions.create 9L (BatchMax(Some 3))
        let batch = Collation.batch tok [ "a"; "a b c d" ] options torch.CPU
        batch.InputIds.shape |> should equal [| 2L; 3L |]
        batch.AttentionMask.shape |> should equal [| 2L; 3L |]
        batch.Lengths |> should equal [| 1L; 4L |]

        let ids = batch.InputIds.data<int64>().ToArray()
        ids[0..2] |> should equal [| 0L; 9L; 9L |]
        ids[3..5] |> should equal [| 0L; 1L; 2L |]

        let mask = batch.AttentionMask.data<bool>().ToArray()
        mask[0..2] |> should equal [| true; false; false |]
    finally
        System.IO.File.Delete(path)

[<Fact>]
let ``Collation.batch without maxLength pads to the untruncated batch max`` () =
    let path = writeVocabFile [| "a"; "b"; "c"; "d"; "[UNK]" |]

    try
        let tok =
            Tokenizer.fromWordPiece {
                WordPieceConfig.create path with
                    SpecialTokens = [ "[UNK]", 4 ]
            }

        let options = CollationOptions.create 9L (BatchMax None)
        let batch = Collation.batch tok [ "a"; "a b c d" ] options torch.CPU
        batch.InputIds.shape |> should equal [| 2L; 4L |]
        batch.Lengths |> should equal [| 1L; 4L |]

        batch.InputIds.data<int64>().ToArray()[0..3]
        |> should equal [| 0L; 9L; 9L; 9L |]
    finally
        System.IO.File.Delete(path)

// --- BERT tokenizer ---

[<Fact>]
let ``fromBert adds CLS and SEP by default`` () =
    let path =
        writeVocabFile [| "[PAD]"; "[UNK]"; "[CLS]"; "[SEP]"; "[MASK]"; "hello" |]

    try
        let tok = Tokenizer.fromBert (BertConfig.create path)
        let ids = tok.encode "hello"
        ids.Head |> should equal 2L
        ids |> List.last |> should equal 3L
        ids |> should equal [ 2L; 5L; 3L ]
    finally
        System.IO.File.Delete(path)

[<Fact>]
let ``fromBert can omit special tokens`` () =
    let path =
        writeVocabFile [| "[PAD]"; "[UNK]"; "[CLS]"; "[SEP]"; "[MASK]"; "hello" |]

    try
        let tok =
            Tokenizer.fromBert {
                BertConfig.create path with
                    AddSpecialTokens = false
            }

        tok.encode "hello" |> should equal [ 5L ]
    finally
        System.IO.File.Delete(path)

[<Fact>]
let ``fromBert splits CJK characters`` () =
    let path =
        writeVocabFile [| "[PAD]"; "[UNK]"; "[CLS]"; "[SEP]"; "[MASK]"; "液"; "晶" |]

    try
        let tok = Tokenizer.fromBert (BertConfig.create path)
        tok.encode "液晶" |> should equal [ 2L; 5L; 6L; 3L ]
    finally
        System.IO.File.Delete(path)

[<Fact>]
let ``fromFunctions uses the provided encode and decode`` () =
    let tok =
        Tokenizer.fromFunctions (fun text -> text |> Seq.map int64 |> Seq.toList) (fun ids ->
            ids |> List.map char |> Array.ofList |> System.String)

    tok.encode "ab" |> should equal [ 97L; 98L ]
    tok.decode [ 97L; 98L ] |> should equal "ab"
    tok.countTokens "ab" |> should equal 2

[<Fact>]
let ``SentencePieceConfig.create matches Create stream defaults`` () =
    let config = SentencePieceConfig.create "spiece.model"
    config.ModelPath |> should equal "spiece.model"
    config.AddBos |> should equal true
    config.AddEos |> should equal false
    config.AddDummyPrefix |> should equal false

[<Fact>]
let ``SentencePieceConfig can enable dummy prefix without BOS or EOS`` () =
    let config = {
        SentencePieceConfig.create "spiece.model" with
            AddBos = false
            AddEos = false
            AddDummyPrefix = true
    }

    config.AddBos |> should equal false
    config.AddEos |> should equal false
    config.AddDummyPrefix |> should equal true
