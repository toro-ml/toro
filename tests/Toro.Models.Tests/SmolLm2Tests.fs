module SmolLm2Tests

open TorchSharp
open Toro
open Toro.Models
open Toro.NN
open Xunit

let private config = {
    VocabSize = 32L
    HiddenSize = 8L
    IntermediateSize = 16L
    NumHiddenLayers = 2
    NumAttentionHeads = 2L
    NumKeyValueHeads = 1L
    MaxPositionEmbeddings = 16L
    RmsNormEps = 1e-5
    RopeTheta = 10000.0
    BosTokenId = 1L
    EosTokenId = 2L
}

[<Fact>]
let ``descriptor uses external canonical names`` () =
    let model = SmolLm2.create config torch.float32 torch.CPU

    try
        let names =
            model
            |> SmolLm2.state
            |> ModelState.namedParams
            |> List.map _.Name

        Assert.Equal("model.embed_tokens.weight", names.Head)
        Assert.Contains("model.layers.1.self_attn.q_proj.weight", names)
        Assert.Equal("model.norm.weight", names |> List.last)
        Assert.Equal(20, names.Length)
    finally
        SmolLm2.dispose model

[<Fact>]
let ``cached decode matches full sequence forward`` () =
    torch.manual_seed 42L |> ignore
    let model = SmolLm2.create config torch.float32 torch.CPU
    use cache = SmolLm2.createCache 1L 4L model

    try
        let matches =
            Toro.inferenceMode (fun () ->
                scoped {
                    let fullIds =
                        (torch.tensor ([| 1L; 5L; 7L; 9L |], dtype = torch.int64)).unsqueeze (0L)

                    let full =
                        model
                        |> SmolLm2.forward {
                            InputIds = fullIds
                            AttentionMask = None
                            PositionIds = None
                            Cache = None
                        }

                    let promptIds =
                        (torch.tensor ([| 1L; 5L; 7L |], dtype = torch.int64)).unsqueeze (0L)

                    model
                    |> SmolLm2.forward {
                        InputIds = promptIds
                        AttentionMask = None
                        PositionIds = None
                        Cache = Some cache
                    }
                    |> ignore

                    let nextId = (torch.tensor ([| 9L |], dtype = torch.int64)).unsqueeze (0L)

                    let decoded =
                        model
                        |> SmolLm2.forward {
                            InputIds = nextId
                            AttentionMask = None
                            PositionIds = None
                            Cache = Some cache
                        }

                    let expected = full.Logits.at [ I 0; I -1 ]
                    let actual = decoded.Logits.at [ I 0; I -1 ]
                    return torch.allclose (expected, actual, rtol = 1e-4, atol = 1e-5)
                })

        Assert.True matches
        Assert.Equal(4L, cache.Length)
    finally
        SmolLm2.dispose model
