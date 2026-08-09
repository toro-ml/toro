module TextTests

open Xunit
open FsUnit.Xunit
open Toro
open Toro.Text
open TestHelper

// --- CharTokenizer tests ---

[<Fact>]
let ``CharTokenizer round-trips encode and decode`` () =
    let tok = CharTokenizer.fromCorpus "hello world"
    let encoded = (tok :> ITokenizer).encode "hello"
    let decoded = (tok :> ITokenizer).decode encoded
    decoded |> should equal "hello"

[<Fact>]
let ``CharTokenizer vocabSize matches unique chars`` () =
    let tok = CharTokenizer.fromCorpus "aabbc"
    (tok :> ITokenizer).vocabSize |> should equal 3

[<Fact>]
let ``CharTokenizer assigns sorted indices`` () =
    let tok = CharTokenizer.fromCorpus "cba"
    let ids = (tok :> ITokenizer).encode "abc"
    ids |> should equal [ 0; 1; 2 ]

// --- Preprocessing tests ---

[<Fact>]
let ``padOrTruncate pads short sequences`` () =
    let result = Preprocessing.padOrTruncate 5 0 [ 1; 2; 3 ]
    result |> should equal [ 1; 2; 3; 0; 0 ]

[<Fact>]
let ``padOrTruncate truncates long sequences`` () =
    let result = Preprocessing.padOrTruncate 3 0 [ 1; 2; 3; 4; 5 ]
    result |> should equal [ 1; 2; 3 ]

[<Fact>]
let ``padOrTruncate preserves exact-length sequences`` () =
    let result = Preprocessing.padOrTruncate 3 0 [ 1; 2; 3 ]
    result |> should equal [ 1; 2; 3 ]

[<Fact>]
let ``batchToTensor produces correct shape`` () =
    let batch = [ [ 1; 2; 3 ]; [ 4; 5 ] ]
    let t = Preprocessing.batchToTensor 4 0 batch I64 Cpu |> unwrap
    t.Shape |> should equal [ 2; 4 ]

[<Fact>]
let ``attentionMask marks pad positions as zero`` () =
    let data: int64 array = [| 1L; 2L; 0L; 0L |]
    let t = Tensor.ofArray (data, Cpu) |> unwrap

    let mask = Preprocessing.attentionMask t 0
    let m0 = mask.at [ I 0 ] |> scalarF32
    let m1 = mask.at [ I 1 ] |> scalarF32
    let m2 = mask.at [ I 2 ] |> scalarF32
    let m3 = mask.at [ I 3 ] |> scalarF32
    m0 |> should (equalWithin 1e-5) 1.0f
    m1 |> should (equalWithin 1e-5) 1.0f
    m2 |> should (equalWithin 1e-5) 0.0f
    m3 |> should (equalWithin 1e-5) 0.0f
