namespace Toro.Models

open System
open TorchSharp
open Toro
open Toro.Models
open Toro.NN

/// A DistilGPT-2 causal language model with tied input and output embeddings.
type DistilGpt2 = {
    Config: DistilGpt2Config
    TokenEmbedding: Embedding
    PositionEmbedding: Embedding
    Blocks: DistilGpt2Block list
    FinalNorm: LayerNorm
}

/// Construction, state, cache, and forward operations for DistilGPT-2.
module DistilGpt2 =

    let private parameter shape dtype device initializer : Tensor =
        Init.toParam shape dtype device initializer

    let private embedding size hiddenSize dtype device : Embedding = {
        Embeddings = parameter [| size; hiddenSize |] dtype device (Init.Randn(0.0, 0.02))
        HiddenSize = hiddenSize
    }

    let private conv1d inputSize outputSize dtype device : DistilGpt2Conv1D = {
        Weight = parameter [| inputSize; outputSize |] dtype device (Init.Randn(0.0, 0.02))
        Bias = parameter [| outputSize |] dtype device (Init.Const 0.0)
    }

    let private norm (config: DistilGpt2Config) dtype device =
        LayerNorm.init
            config.EmbeddingSize
            {
                LayerNormConfig.defaultConfig with
                    Eps = config.LayerNormEps
            }
            dtype
            device

    let private block (config: DistilGpt2Config) dtype device : DistilGpt2Block = {
        Norm1 = norm config dtype device
        Attention = {
            Qkv = conv1d config.EmbeddingSize (3L * config.EmbeddingSize) dtype device
            Output = conv1d config.EmbeddingSize config.EmbeddingSize dtype device
        }
        Norm2 = norm config dtype device
        Mlp = {
            Input = conv1d config.EmbeddingSize config.IntermediateSize dtype device
            Output = conv1d config.IntermediateSize config.EmbeddingSize dtype device
        }
    }

    let private bias (layer: LayerNorm) =
        layer.Bias
        |> Option.defaultWith (fun () -> invalidOp "DistilGPT-2 LayerNorm must contain bias.")

    let private namedParameters (model: DistilGpt2) =
        seq {
            yield "transformer.wte.weight", model.TokenEmbedding.Embeddings
            yield "transformer.wpe.weight", model.PositionEmbedding.Embeddings

            for index, layer in List.indexed model.Blocks do
                let prefix = $"transformer.h.{index}"
                yield $"{prefix}.ln_1.weight", layer.Norm1.Weight
                yield $"{prefix}.ln_1.bias", bias layer.Norm1
                yield $"{prefix}.attn.c_attn.weight", layer.Attention.Qkv.Weight
                yield $"{prefix}.attn.c_attn.bias", layer.Attention.Qkv.Bias
                yield $"{prefix}.attn.c_proj.weight", layer.Attention.Output.Weight
                yield $"{prefix}.attn.c_proj.bias", layer.Attention.Output.Bias
                yield $"{prefix}.ln_2.weight", layer.Norm2.Weight
                yield $"{prefix}.ln_2.bias", bias layer.Norm2
                yield $"{prefix}.mlp.c_fc.weight", layer.Mlp.Input.Weight
                yield $"{prefix}.mlp.c_fc.bias", layer.Mlp.Input.Bias
                yield $"{prefix}.mlp.c_proj.weight", layer.Mlp.Output.Weight
                yield $"{prefix}.mlp.c_proj.bias", layer.Mlp.Output.Bias

            yield "transformer.ln_f.weight", model.FinalNorm.Weight
            yield "transformer.ln_f.bias", bias model.FinalNorm
        }

    /// State descriptor whose canonical names and shapes match Hugging Face DistilGPT-2 weights.
    let descriptor: ModelDescriptor<DistilGpt2> = {
        NamedParameters = namedParameters
        NamedBuffers = fun _ -> Seq.empty
        Dispose = TensorOwner.disposeDistinct namedParameters
    }

    /// Create a DistilGPT-2 model from a validated configuration.
    let create (config: DistilGpt2Config) (dtype: torch.ScalarType) (device: torch.Device) : DistilGpt2 =
        DistilGpt2Config.validate config

        {
            Config = config
            TokenEmbedding = embedding config.VocabSize config.EmbeddingSize dtype device
            PositionEmbedding = embedding config.MaxPositionEmbeddings config.EmbeddingSize dtype device
            Blocks = List.init config.NumHiddenLayers (fun _ -> block config dtype device)
            FinalNorm = norm config dtype device
        }

    /// Create a validated named state view using Hugging Face weight names.
    let state (model: DistilGpt2) =
        Toro.NN.Model.stateWith descriptor model

    /// Dispose tensors owned by a DistilGPT-2 model.
    let dispose (model: DistilGpt2) =
        ModelDescriptor.dispose descriptor model

    /// Allocate a reusable fixed-capacity key/value cache for a model.
    let createCache batchSize capacity (model: DistilGpt2) =
        let parameter = model.TokenEmbedding.Embeddings
        new DistilGpt2Cache(model.Config, batchSize, capacity, parameter.dtype, parameter.device)

    let private geluNew (input: Tensor) =
        let cubic = input.pow (scalar 3.0)
        let inner = input + cubic * scalar 0.044715
        let cdf = (inner * scalar (sqrt (2.0 / Math.PI))).tanh () + scalar 1.0
        input * scalar 0.5 * cdf

    let private attentionForward
        config
        layerIndex
        (attention: DistilGpt2Attention)
        (input: Tensor)
        mask
        isCausal
        cache
        cacheStart
        =
        let batchSize = input.shape[0]
        let sequenceLength = input.shape[1]
        let headSize = config.EmbeddingSize / config.NumAttentionHeads
        let chunks = attention.Qkv.forward(input).chunk (3L, dim = -1L)

        let toHeads (tensor: Tensor) =
            tensor.reshape([| batchSize; sequenceLength; config.NumAttentionHeads; headSize |]).permute ([| 0L; 2L; 1L; 3L |])

        let query = toHeads chunks[0]
        let key = toHeads chunks[1]
        let value = toHeads chunks[2]

        let key, value =
            match cache with
            | Some(cache: DistilGpt2Cache) -> cache.Append(layerIndex, cacheStart, key, value)
            | None -> key, value

        let mask = mask |> Option.defaultValue null

        let attended =
            torch.nn.functional.scaled_dot_product_attention (query, key, value, attn_mask = mask, is_casual = isCausal)

        attended.permute([| 0L; 2L; 1L; 3L |]).contiguous().reshape ([| batchSize; sequenceLength; config.EmbeddingSize |])
        |> attention.Output.forward

    let private blockForward config layerIndex (block: DistilGpt2Block) input mask isCausal cache cacheStart =
        let attended =
            input
            |> block.Norm1.forward
            |> fun hidden -> attentionForward config layerIndex block.Attention hidden mask isCausal cache cacheStart

        let hidden = input + attended

        let projected =
            hidden
            |> block.Norm2.forward
            |> block.Mlp.Input.forward
            |> geluNew
            |> block.Mlp.Output.forward

        hidden + projected

    /// Run DistilGPT-2. When a cache is supplied, only new tokens should be passed after prefill.
    let forward (input: DistilGpt2Input) (model: DistilGpt2) : DistilGpt2Output =
        let prepared =
            CausalInput.prepare
                "DistilGPT-2"
                model.Config.MaxPositionEmbeddings
                (fun (cache: DistilGpt2Cache) -> cache.Length)
                (fun (cache: DistilGpt2Cache) batchSize sequenceLength -> cache.Validate(batchSize, sequenceLength))
                input

        let hidden =
            model.TokenEmbedding.forward input.InputIds
            + model.PositionEmbedding.forward prepared.PositionIds

        let hidden =
            (hidden, List.indexed model.Blocks)
            ||> List.fold (fun state (index, layer) ->
                blockForward
                    model.Config
                    index
                    layer
                    state
                    prepared.AttentionMask
                    prepared.IsCausal
                    input.Cache
                    prepared.CacheStart)

        let hidden = model.FinalNorm.forward hidden
        let logits = hidden.matmul (model.TokenEmbedding.Embeddings.t ())

        input.Cache
        |> Option.iter (fun cache -> cache.Advance prepared.SequenceLength)

        { Logits = logits; Cache = input.Cache }

    /// Bind a DistilGPT-2 model to the common typed causal language-model interface.
    let asCausalLm (model: DistilGpt2) : CausalLm<DistilGpt2Cache> = {
        ContextLength = model.Config.MaxPositionEmbeddings
        EosTokenIds = Set.singleton model.Config.EosTokenId
        Device = model.TokenEmbedding.Embeddings.device
        CreateCache = fun batchSize capacity -> createCache batchSize capacity model
        CacheLength = _.Length
        DisposeCache = fun cache -> (cache :> IDisposable).Dispose()
        Forward = fun input -> forward input model
    }

    /// Load and validate a Hugging Face config.json file.
    let loadConfig (path: string) = DistilGpt2Checkpoint.loadConfig path

    /// Load config and a strict single-file or sharded SafeTensors state from a local directory.
    let loadFromDirectory (directory: string) (device: torch.Device) : DistilGpt2 * LoadReport =
        let configPath, reader = LocalModelAssets.openReader "DistilGPT-2" directory
        use reader = reader
        let config = loadConfig configPath
        let dtype = LocalModelAssets.dtype "DistilGPT-2" "transformer.wte.weight" reader
        let model = create config dtype device
        let mapping = NameMapping.create [ NameRule.ignoreSuffix "attn.bias" ]

        try
            let report = ModelState.loadSafeTensorsWith mapping Strict (state model) reader
            model, report
        with _ ->
            dispose model
            reraise ()
