namespace Toro.Text

/// Tokenizer that maps between strings and integer token sequences.
type ITokenizer =
    abstract encode: string -> int list
    abstract decode: int list -> string
    abstract vocabSize: int

/// Character-level tokenizer built from a corpus.
type CharTokenizer = {
    CharToId: Map<char, int>
    IdToChar: Map<int, char>
} with

    interface ITokenizer with
        member this.encode text =
            text |> Seq.map (fun c -> this.CharToId[c]) |> Seq.toList

        member this.decode ids =
            ids
            |> List.map (fun id -> this.IdToChar[id])
            |> List.toArray
            |> System.String

        member this.vocabSize = this.CharToId.Count

module CharTokenizer =
    /// Build a character tokenizer from the unique characters in the corpus.
    let fromCorpus (text: string) : CharTokenizer =
        let chars = text |> Seq.distinct |> Seq.sort |> Seq.toArray

        let charToId = chars |> Array.mapi (fun i c -> c, i) |> Map.ofArray

        let idToChar = chars |> Array.mapi (fun i c -> i, c) |> Map.ofArray

        {
            CharToId = charToId
            IdToChar = idToChar
        }
