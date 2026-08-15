namespace Toro.Models

open System
open System.IO
open System.Text.Json
open TorchSharp
open Toro
open Toro.NN

/// Configuration values used by a DistilGPT-2 causal language model.
type DistilGpt2Config = {
    VocabSize: int64
    EmbeddingSize: int64
    IntermediateSize: int64
    NumHiddenLayers: int
    NumAttentionHeads: int64
    MaxPositionEmbeddings: int64
    LayerNormEps: float
    BosTokenId: int64
    EosTokenId: int64
}

/// Validation operations for DistilGPT-2 configurations.
module DistilGpt2Config =

    /// Validate dimensions and constants required by the implemented architecture.
    let validate (config: DistilGpt2Config) =
        if
            config.VocabSize <= 0L
            || config.EmbeddingSize <= 0L
            || config.IntermediateSize <= 0L
            || config.NumHiddenLayers <= 0
            || config.NumAttentionHeads <= 0L
            || config.MaxPositionEmbeddings <= 0L
            || config.LayerNormEps <= 0.0
            || not (Double.IsFinite config.LayerNormEps)
        then
            invalidArg (nameof config) "DistilGPT-2 dimensions and constants must be finite and positive."

        if config.EmbeddingSize % config.NumAttentionHeads <> 0L then
            invalidArg (nameof config) "Embedding size must be divisible by the number of attention heads."

        if
            config.BosTokenId < 0L
            || config.BosTokenId >= config.VocabSize
        then
            invalidArg (nameof config) "BOS token ID must be within the vocabulary."

        if
            config.EosTokenId < 0L
            || config.EosTokenId >= config.VocabSize
        then
            invalidArg (nameof config) "EOS token ID must be within the vocabulary."

/// A GPT-2 Conv1D projection stored in Hugging Face [input, output] weight layout.
type Gpt2Conv1D = {
    Weight: Tensor
    Bias: Tensor
} with

    /// Apply the affine projection without transposing its stored weight.
    member this.forward(input: Tensor) = input.matmul this.Weight + this.Bias

/// Projection layers in one DistilGPT-2 self-attention block.
type DistilGpt2Attention = { Qkv: Gpt2Conv1D; Output: Gpt2Conv1D }

/// Projection layers in one DistilGPT-2 feed-forward block.
type DistilGpt2Mlp = {
    Input: Gpt2Conv1D
    Output: Gpt2Conv1D
}

/// One DistilGPT-2 transformer block.
type DistilGpt2Block = {
    Norm1: LayerNorm
    Attention: DistilGpt2Attention
    Norm2: LayerNorm
    Mlp: DistilGpt2Mlp
}

/// A DistilGPT-2 causal language model with tied input and output embeddings.
type DistilGpt2 = {
    Config: DistilGpt2Config
    TokenEmbedding: Embedding
    PositionEmbedding: Embedding
    Blocks: DistilGpt2Block list
    FinalNorm: LayerNorm
}

/// A fixed-capacity, per-layer DistilGPT-2 key/value cache.
type DistilGpt2Cache internal (config: DistilGpt2Config, batchSize: int64, capacity: int64, dtype, device) =
    do
        DistilGpt2Config.validate config

        if batchSize <= 0L then
            invalidArg (nameof batchSize) "Cache batch size must be positive."

        if capacity <= 0L || capacity > config.MaxPositionEmbeddings then
            invalidArg (nameof capacity) $"Cache capacity must be between 1 and {config.MaxPositionEmbeddings}."

    let headSize = config.EmbeddingSize / config.NumAttentionHeads

    let storage =
        new FixedKvCache(
            nameof DistilGpt2Cache,
            config.NumHiddenLayers,
            batchSize,
            config.NumAttentionHeads,
            capacity,
            headSize,
            dtype,
            device
        )

    /// Number of batch items stored by this cache.
    member _.BatchSize = storage.BatchSize

    /// Maximum number of tokens stored by this cache.
    member _.Capacity = storage.Capacity

    /// Number of tokens currently stored by this cache.
    member _.Length = storage.Length

    /// Remove all logical entries without reallocating storage.
    member _.Reset() = storage.Reset()

    member internal _.Validate(batch: int64, sequenceLength: int64) = storage.Validate(batch, sequenceLength)

    member internal _.Append(layerIndex: int, start: int64, key: Tensor, value: Tensor) =
        storage.Append(layerIndex, start, key, value)

    member internal _.Advance(sequenceLength: int64) = storage.Advance sequenceLength

    interface IDisposable with
        member _.Dispose() = (storage :> IDisposable).Dispose()

/// Tensor inputs accepted by DistilGPT-2.
type DistilGpt2Input = CausalLmInput<DistilGpt2Cache>

/// Tensor outputs produced by DistilGPT-2.
type DistilGpt2Output = CausalLmOutput<DistilGpt2Cache>

module private DistilGpt2ConfigJson =

    let private label = "DistilGPT-2"
    let private tryProperty root name = JsonConfig.tryProperty root name

    let private int64Element name value =
        JsonConfig.int64Element label name value

    let private int64Value root name = JsonConfig.int64Value label root name
    let private floatValue root name = JsonConfig.floatValue label root name
    let private stringValue root name = JsonConfig.stringValue label root name

    let load (path: string) =
        use document = JsonDocument.Parse(File.ReadAllText path)
        let root = document.RootElement

        JsonConfig.validateObject label root

        if stringValue root "model_type" <> "gpt2" then
            invalidOp "DistilGPT-2 config 'model_type' must be 'gpt2'."

        if stringValue root "activation_function" <> "gelu_new" then
            invalidOp "DistilGPT-2 config 'activation_function' must be 'gelu_new'."

        let embeddingSize = int64Value root "n_embd"
        let maxPositions = int64Value root "n_positions"

        if int64Value root "n_ctx" <> maxPositions then
            invalidOp "DistilGPT-2 n_ctx must equal n_positions."

        let intermediateSize =
            match tryProperty root "n_inner" with
            | Some value when value.ValueKind <> JsonValueKind.Null -> int64Element "n_inner" value
            | _ -> 4L * embeddingSize

        let config = {
            VocabSize = int64Value root "vocab_size"
            EmbeddingSize = embeddingSize
            IntermediateSize = intermediateSize
            NumHiddenLayers = int (int64Value root "n_layer")
            NumAttentionHeads = int64Value root "n_head"
            MaxPositionEmbeddings = maxPositions
            LayerNormEps = floatValue root "layer_norm_epsilon"
            BosTokenId = int64Value root "bos_token_id"
            EosTokenId = int64Value root "eos_token_id"
        }

        DistilGpt2Config.validate config
        config

/// Construction, state, loading, cache, and forward operations for DistilGPT-2.
module DistilGpt2 =

    let private parameter shape dtype device initializer =
        Init.toParam shape dtype device initializer

    let private embedding size hiddenSize dtype device = {
        Embeddings = parameter [| size; hiddenSize |] dtype device (Init.Randn(0.0, 0.02))
        HiddenSize = hiddenSize
    }

    let private conv1d inputSize outputSize dtype device = {
        Weight = parameter [| inputSize; outputSize |] dtype device (Init.Randn(0.0, 0.02))
        Bias = parameter [| outputSize |] dtype device (Init.Const 0.0)
    }

    let private norm config dtype device =
        LayerNorm.init
            config.EmbeddingSize
            {
                LayerNormConfig.defaultConfig with
                    Eps = config.LayerNormEps
            }
            dtype
            device

    let private block config dtype device = {
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
    let state (model: DistilGpt2) = Model.stateWith descriptor model

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

    let private blockForward config layerIndex block input mask isCausal cache cacheStart =
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

    /// Load config and a strict single-file or sharded SafeTensors state from a local directory.
    let loadFromDirectory (directory: string) (device: torch.Device) : DistilGpt2 * LoadReport =
        let configPath, reader = LocalModelAssets.openReader "DistilGPT-2" directory
        use reader = reader
        let config = DistilGpt2ConfigJson.load configPath
        let dtype = LocalModelAssets.dtype "DistilGPT-2" "transformer.wte.weight" reader
        let model = create config dtype device
        let mapping = NameMapping.create [ NameRule.ignoreSuffix "attn.bias" ]

        try
            let report = ModelState.loadSafeTensorsWith mapping Strict (state model) reader
            model, report
        with _ ->
            dispose model
            reraise ()
