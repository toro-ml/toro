module DistilGpt2Tests

open TorchSharp
open Toro
open Toro.Models
open Toro.NN
open Xunit

let private config: DistilGpt2Config = {
    VocabSize = 32L
    EmbeddingSize = 8L
    IntermediateSize = 16L
    NumHiddenLayers = 2
    NumAttentionHeads = 2L
    MaxPositionEmbeddings = 16L
    LayerNormEps = 1e-5
    BosTokenId = 1L
    EosTokenId = 2L
}

[<Fact>]
let ``descriptor preserves Hugging Face Conv1D shapes`` () =
    let model = DistilGpt2.create config torch.float32 torch.CPU

    try
        let parameters = model |> DistilGpt2.state |> ModelState.namedParams

        let qkv =
            parameters
            |> List.find (fun item -> item.Name = "transformer.h.0.attn.c_attn.weight")

        Assert.Equal<int64>(config.EmbeddingSize, qkv.Tensor.shape[0])
        Assert.Equal<int64>(3L * config.EmbeddingSize, qkv.Tensor.shape[1])
        Assert.Equal(28, parameters.Length)
    finally
        DistilGpt2.dispose model

[<Fact>]
let ``cached decode matches full sequence forward`` () =
    torch.manual_seed 42L |> ignore
    let model = DistilGpt2.create config torch.float32 torch.CPU
    use cache = DistilGpt2.createCache 1L 4L model

    try
        let matches =
            Toro.inferenceMode (fun () ->
                scoped {
                    let fullIds =
                        (torch.tensor ([| 1L; 5L; 7L; 9L |], dtype = torch.int64)).unsqueeze (0L)

                    let full =
                        model
                        |> DistilGpt2.forward {
                            InputIds = fullIds
                            AttentionMask = None
                            PositionIds = None
                            Cache = None
                        }

                    let promptIds =
                        (torch.tensor ([| 1L; 5L; 7L |], dtype = torch.int64)).unsqueeze (0L)

                    model
                    |> DistilGpt2.forward {
                        InputIds = promptIds
                        AttentionMask = None
                        PositionIds = None
                        Cache = Some cache
                    }
                    |> ignore

                    let nextId = (torch.tensor ([| 9L |], dtype = torch.int64)).unsqueeze (0L)

                    let decoded =
                        model
                        |> DistilGpt2.forward {
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
        DistilGpt2.dispose model

[<Fact>]
let ``checkpoint loader uses the family model descriptor`` () =
    let directory =
        System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"toro-distilgpt2-{System.Guid.NewGuid():N}")

    System.IO.Directory.CreateDirectory(directory) |> ignore
    let source = DistilGpt2.create config torch.float32 torch.CPU

    try
        let configJson =
            """{"model_type":"gpt2","activation_function":"gelu_new","n_embd":8,"n_positions":16,"n_ctx":16,"n_inner":16,"vocab_size":32,"n_layer":2,"n_head":2,"layer_norm_epsilon":0.00001,"bos_token_id":1,"eos_token_id":2}"""

        System.IO.File.WriteAllText(System.IO.Path.Combine(directory, "config.json"), configJson)

        ModelState.save (DistilGpt2.state source) (System.IO.Path.Combine(directory, "model.safetensors"))

        let loaded, report = DistilGpt2.loadFromDirectory directory torch.CPU

        try
            Assert.Equal(28, report.Loaded.Length)
        finally
            DistilGpt2.dispose loaded
    finally
        DistilGpt2.dispose source
        System.IO.Directory.Delete(directory, true)
