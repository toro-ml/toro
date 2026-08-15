namespace Toro.Models

open System
open System.Collections.Generic
open System.IO
open System.Text.Json
open TorchSharp
open Toro
open Toro.NN

/// Configuration values used by a SmolLM2 causal language model.
type SmolLm2Config = {
    VocabSize: int64
    HiddenSize: int64
    IntermediateSize: int64
    NumHiddenLayers: int
    NumAttentionHeads: int64
    NumKeyValueHeads: int64
    MaxPositionEmbeddings: int64
    RmsNormEps: float
    RopeTheta: float
    BosTokenId: int64
    EosTokenId: int64
}

/// Validation operations for SmolLM2 configurations.
module SmolLm2Config =

    /// Validate dimensions and constants required by the implemented architecture.
    let validate (config: SmolLm2Config) =
        if
            config.VocabSize <= 0L
            || config.HiddenSize <= 0L
            || config.IntermediateSize <= 0L
            || config.NumHiddenLayers <= 0
            || config.NumAttentionHeads <= 0L
            || config.NumKeyValueHeads <= 0L
            || config.MaxPositionEmbeddings <= 0L
            || config.RmsNormEps <= 0.0
            || not (Double.IsFinite config.RmsNormEps)
            || config.RopeTheta <= 0.0
            || not (Double.IsFinite config.RopeTheta)
        then
            invalidArg (nameof config) "SmolLM2 dimensions and numeric constants must be finite and positive."

        if config.HiddenSize % config.NumAttentionHeads <> 0L then
            invalidArg (nameof config) "SmolLM2 hidden size must be divisible by the number of attention heads."

        let headSize = config.HiddenSize / config.NumAttentionHeads

        if headSize % 2L <> 0L then
            invalidArg (nameof config) "SmolLM2 attention head size must be even for rotary embedding."

        if config.NumAttentionHeads % config.NumKeyValueHeads <> 0L then
            invalidArg (nameof config) "SmolLM2 attention heads must be divisible by key/value heads."

        if
            config.BosTokenId < 0L
            || config.BosTokenId >= config.VocabSize
        then
            invalidArg (nameof config) "SmolLM2 BOS token ID must be within the vocabulary."

        if
            config.EosTokenId < 0L
            || config.EosTokenId >= config.VocabSize
        then
            invalidArg (nameof config) "SmolLM2 EOS token ID must be within the vocabulary."

/// Projection layers in one SmolLM2 grouped-query attention block.
type SmolLm2Attention = {
    Query: Linear
    Key: Linear
    Value: Linear
    Output: Linear
}

/// Projection layers in one SmolLM2 SwiGLU feed-forward block.
type SmolLm2Mlp = {
    Gate: Linear
    Up: Linear
    Down: Linear
}

/// One SmolLM2 transformer block.
type SmolLm2Block = {
    InputNorm: RmsNorm
    Attention: SmolLm2Attention
    PostAttentionNorm: RmsNorm
    Mlp: SmolLm2Mlp
}

/// A SmolLM2 causal language model with tied input and output embeddings.
type SmolLm2 = {
    Config: SmolLm2Config
    TokenEmbedding: Embedding
    Blocks: SmolLm2Block list
    FinalNorm: RmsNorm
}

/// A fixed-capacity, per-layer SmolLM2 key/value cache.
type SmolLm2Cache internal (config: SmolLm2Config, batchSize: int64, capacity: int64, dtype, device) =
    let validated =
        SmolLm2Config.validate config

        if batchSize <= 0L then
            invalidArg (nameof batchSize) "Cache batch size must be positive."

        if capacity <= 0L || capacity > config.MaxPositionEmbeddings then
            invalidArg (nameof capacity) $"Cache capacity must be between 1 and {config.MaxPositionEmbeddings}."

    let headSize = config.HiddenSize / config.NumAttentionHeads

    let keys =
        Array.init config.NumHiddenLayers (fun _ ->
            torch.empty ([| batchSize; config.NumKeyValueHeads; capacity; headSize |], dtype = dtype, device = device))

    let values =
        Array.init config.NumHiddenLayers (fun _ ->
            torch.empty ([| batchSize; config.NumKeyValueHeads; capacity; headSize |], dtype = dtype, device = device))

    let mutable length = 0L
    let mutable disposed = false

    let ensureAvailable () =
        if disposed then
            raise (ObjectDisposedException(nameof SmolLm2Cache))

    do ignore validated

    /// Number of batch items stored by this cache.
    member _.BatchSize = batchSize

    /// Maximum number of tokens stored by this cache.
    member _.Capacity = capacity

    /// Number of tokens currently stored by this cache.
    member _.Length = length

    /// Remove all logical entries without reallocating storage.
    member _.Reset() =
        ensureAvailable ()
        length <- 0L

    member internal _.Validate(batch: int64, sequenceLength: int64) =
        ensureAvailable ()

        if batch <> batchSize then
            invalidArg (nameof batch) $"Cache batch size is {batchSize}, but input batch size is {batch}."

        if length + sequenceLength > capacity then
            invalidOp $"KV cache capacity {capacity} is too small for {length + sequenceLength} tokens."

    member internal _.Append(layerIndex: int, start: int64, key: Tensor, value: Tensor) =
        ensureAvailable ()

        if start <> length then
            invalidOp $"KV cache append started at {start}, but the current length is {length}."

        let sequenceLength = key.shape[2]
        use keyDestination = keys[layerIndex].narrow (2L, start, sequenceLength)
        use valueDestination = values[layerIndex].narrow (2L, start, sequenceLength)
        keyDestination.copyInPlace key
        valueDestination.copyInPlace value
        keys[layerIndex].narrow (2L, 0L, start + sequenceLength), values[layerIndex].narrow (2L, 0L, start + sequenceLength)

    member internal _.Advance(sequenceLength: int64) =
        ensureAvailable ()
        length <- length + sequenceLength

    interface IDisposable with
        member _.Dispose() =
            if not disposed then
                disposed <- true
                keys |> Array.iter _.Dispose()
                values |> Array.iter _.Dispose()

/// Tensor inputs accepted by SmolLM2.
type SmolLm2Input = {
    /// Token IDs with shape [batch, sequence].
    InputIds: Tensor
    /// Optional 0/1 padding mask with shape [batch, total sequence].
    AttentionMask: Tensor option
    /// Optional absolute positions with shape [sequence] or [batch, sequence].
    PositionIds: Tensor option
    /// Optional cache used for prefill and incremental decoding.
    Cache: SmolLm2Cache option
}

/// Tensor outputs produced by SmolLM2.
type SmolLm2Output = {
    /// Vocabulary logits with shape [batch, sequence, vocabulary].
    Logits: Tensor
    /// The cache supplied in the input, after the successful forward pass.
    Cache: SmolLm2Cache option
}

module private ConfigJson =

    let private property (root: JsonElement) (name: string) : JsonElement =
        match root.TryGetProperty name with
        | true, value -> value
        | false, _ -> invalidOp $"SmolLM2 config is missing '{name}'."

    let private int64Value (root: JsonElement) (name: string) =
        let value = property root name

        match value.TryGetInt64() with
        | true, result -> result
        | false, _ -> invalidOp $"SmolLM2 config '{name}' must be an integer."

    let private floatValue (root: JsonElement) (name: string) =
        let value = property root name

        if value.ValueKind <> JsonValueKind.Number then
            invalidOp $"SmolLM2 config '{name}' must be a number."

        value.GetDouble()

    let private boolValue (root: JsonElement) (name: string) =
        let value = property root name

        match value.ValueKind with
        | JsonValueKind.True -> true
        | JsonValueKind.False -> false
        | _ -> invalidOp $"SmolLM2 config '{name}' must be a boolean."

    let private stringValue (root: JsonElement) (name: string) =
        let value = property root name

        if value.ValueKind <> JsonValueKind.String then
            invalidOp $"SmolLM2 config '{name}' must be a string."

        value.GetString()

    let load (path: string) =
        use document = JsonDocument.Parse(File.ReadAllText path)
        let root = document.RootElement

        if root.ValueKind <> JsonValueKind.Object then
            invalidOp "SmolLM2 config must be a JSON object."

        let duplicates =
            root.EnumerateObject()
            |> Seq.groupBy _.Name
            |> Seq.tryFind (fun (_, values) -> Seq.length values > 1)

        match duplicates with
        | Some(name, _) -> invalidOp $"SmolLM2 config contains duplicate key '{name}'."
        | None -> ()

        if stringValue root "model_type" <> "llama" then
            invalidOp "SmolLM2 config 'model_type' must be 'llama'."

        if stringValue root "hidden_act" <> "silu" then
            invalidOp "SmolLM2 config 'hidden_act' must be 'silu'."

        if not (boolValue root "tie_word_embeddings") then
            invalidOp "SmolLM2 requires tied word embeddings."

        if boolValue root "attention_bias" then
            invalidOp "SmolLM2 attention projection bias is not supported."

        if boolValue root "mlp_bias" then
            invalidOp "SmolLM2 MLP projection bias is not supported."

        if floatValue root "attention_dropout" <> 0.0 then
            invalidOp "SmolLM2 attention dropout must be zero."

        if boolValue root "rope_interleaved" then
            invalidOp "SmolLM2 interleaved rotary embedding is not supported."

        let ropeScaling = property root "rope_scaling"

        if ropeScaling.ValueKind <> JsonValueKind.Null then
            invalidOp "SmolLM2 rope scaling is not supported."

        let config = {
            VocabSize = int64Value root "vocab_size"
            HiddenSize = int64Value root "hidden_size"
            IntermediateSize = int64Value root "intermediate_size"
            NumHiddenLayers = int (int64Value root "num_hidden_layers")
            NumAttentionHeads = int64Value root "num_attention_heads"
            NumKeyValueHeads = int64Value root "num_key_value_heads"
            MaxPositionEmbeddings = int64Value root "max_position_embeddings"
            RmsNormEps = floatValue root "rms_norm_eps"
            RopeTheta = floatValue root "rope_theta"
            BosTokenId = int64Value root "bos_token_id"
            EosTokenId = int64Value root "eos_token_id"
        }

        SmolLm2Config.validate config
        config

/// Construction, state, loading, cache, and forward operations for SmolLM2.
module SmolLm2 =

    let private norm config dtype device =
        RmsNorm.init config.HiddenSize config.RmsNormEps dtype device

    let private block config dtype device = {
        InputNorm = norm config dtype device
        Attention = {
            Query = Linear.initNoBias config.HiddenSize config.HiddenSize dtype device
            Key =
                Linear.initNoBias
                    config.HiddenSize
                    (config.NumKeyValueHeads * config.HiddenSize
                     / config.NumAttentionHeads)
                    dtype
                    device
            Value =
                Linear.initNoBias
                    config.HiddenSize
                    (config.NumKeyValueHeads * config.HiddenSize
                     / config.NumAttentionHeads)
                    dtype
                    device
            Output = Linear.initNoBias config.HiddenSize config.HiddenSize dtype device
        }
        PostAttentionNorm = norm config dtype device
        Mlp = {
            Gate = Linear.initNoBias config.HiddenSize config.IntermediateSize dtype device
            Up = Linear.initNoBias config.HiddenSize config.IntermediateSize dtype device
            Down = Linear.initNoBias config.IntermediateSize config.HiddenSize dtype device
        }
    }

    let private namedParameters (model: SmolLm2) =
        seq {
            yield "model.embed_tokens.weight", model.TokenEmbedding.Embeddings

            for index, layer in List.indexed model.Blocks do
                let prefix = $"model.layers.{index}"
                yield $"{prefix}.input_layernorm.weight", layer.InputNorm.Inner.Weight
                yield $"{prefix}.self_attn.q_proj.weight", layer.Attention.Query.Weight
                yield $"{prefix}.self_attn.k_proj.weight", layer.Attention.Key.Weight
                yield $"{prefix}.self_attn.v_proj.weight", layer.Attention.Value.Weight
                yield $"{prefix}.self_attn.o_proj.weight", layer.Attention.Output.Weight
                yield $"{prefix}.post_attention_layernorm.weight", layer.PostAttentionNorm.Inner.Weight
                yield $"{prefix}.mlp.gate_proj.weight", layer.Mlp.Gate.Weight
                yield $"{prefix}.mlp.up_proj.weight", layer.Mlp.Up.Weight
                yield $"{prefix}.mlp.down_proj.weight", layer.Mlp.Down.Weight

            yield "model.norm.weight", model.FinalNorm.Inner.Weight
        }

    /// State descriptor whose canonical names match Hugging Face SmolLM2 weight names.
    let descriptor: ModelDescriptor<SmolLm2> = {
        NamedParameters = namedParameters
        NamedBuffers = fun _ -> Seq.empty
        Dispose =
            fun model ->
                let seen = HashSet<obj>(ReferenceEqualityComparer.Instance)

                for _, tensor in namedParameters model do
                    if seen.Add(box tensor) then
                        tensor.Dispose()
    }

    /// Create a SmolLM2 model from a validated configuration.
    let create (config: SmolLm2Config) (dtype: torch.ScalarType) (device: torch.Device) : SmolLm2 =
        SmolLm2Config.validate config

        {
            Config = config
            TokenEmbedding = Embedding.init config.VocabSize config.HiddenSize dtype device
            Blocks = List.init config.NumHiddenLayers (fun _ -> block config dtype device)
            FinalNorm = norm config dtype device
        }

    /// Create a validated named state view using Hugging Face weight names.
    let state (model: SmolLm2) : ModelState = Model.stateWith descriptor model

    /// Dispose tensors owned by a SmolLM2 model.
    let dispose (model: SmolLm2) =
        ModelDescriptor.dispose descriptor model

    /// Allocate a reusable fixed-capacity key/value cache for a model.
    let createCache (batchSize: int64) (capacity: int64) (model: SmolLm2) : SmolLm2Cache =
        let parameter = model.TokenEmbedding.Embeddings
        new SmolLm2Cache(model.Config, batchSize, capacity, parameter.dtype, parameter.device)

    let private rotateHalf (input: Tensor) =
        let halves = input.chunk (2L, dim = -1L)
        torch.cat ([| halves[1].neg (); halves[0] |], dim = -1L)

    let private rotaryEmbedding (config: SmolLm2Config) (positionIds: Tensor) (dtype: torch.ScalarType) (device: torch.Device) =
        let headSize = config.HiddenSize / config.NumAttentionHeads
        let positions = positionIds.``to``(device).to_type(torch.float32).unsqueeze (-1L)

        let dimensions =
            torch.arange(scalar (float (headSize / 2L)), dtype = torch.float32, device = device).mul (scalar 2.0)

        let inverseFrequencies =
            (dimensions / scalar (float headSize)
             * scalar (log config.RopeTheta))
                .neg()
                .exp ()

        let frequencies = positions * inverseFrequencies
        let embeddings = torch.cat ([| frequencies; frequencies |], dim = -1L)

        let cosines = embeddings.cos().to_type dtype
        let sines = embeddings.sin().to_type dtype

        if positionIds.shape.Length = 1 then
            cosines.unsqueeze(0L).unsqueeze (0L), sines.unsqueeze(0L).unsqueeze (0L)
        else
            cosines.unsqueeze (1L), sines.unsqueeze (1L)

    let private applyRotaryEmbedding (input: Tensor) (cosines: Tensor) (sines: Tensor) =
        input * cosines + rotateHalf input * sines

    let private attentionMask (input: SmolLm2Input) batchSize sequenceLength start totalLength =
        let paddingMask =
            input.AttentionMask
            |> Option.map (fun mask ->
                if mask.shape <> [| batchSize; totalLength |] then
                    invalidArg (nameof input.AttentionMask) $"Attention mask shape must be [{batchSize}, {totalLength}]."

                mask.to_type(torch.ScalarType.Bool).unsqueeze(1L).unsqueeze (1L))

        let needsExplicitCausalMask = start > 0L && sequenceLength > 1L

        match paddingMask, needsExplicitCausalMask with
        | None, false -> None, start = 0L
        | Some padding, false when sequenceLength = 1L -> Some padding, false
        | padding, _ ->
            let keyPositions =
                torch.arange (totalLength, dtype = torch.int64, device = input.InputIds.device)

            let queryPositions =
                torch.arange (start, start + sequenceLength, dtype = torch.int64, device = input.InputIds.device)

            let causal =
                (keyPositions.unsqueeze (0L)
                 .<=. queryPositions.unsqueeze (1L))
                    .unsqueeze(0L)
                    .unsqueeze (0L)

            match padding with
            | Some value -> Some(causal.logical_and value), false
            | None -> Some causal, false

    let private attentionForward
        (config: SmolLm2Config)
        (layerIndex: int)
        (attention: SmolLm2Attention)
        (input: Tensor)
        (cosines: Tensor)
        (sines: Tensor)
        (mask: Tensor option)
        (isCausal: bool)
        (cache: SmolLm2Cache option)
        (cacheStart: int64)
        =
        let batchSize = input.shape[0]
        let sequenceLength = input.shape[1]
        let headSize = config.HiddenSize / config.NumAttentionHeads

        let toHeads heads (tensor: Tensor) =
            tensor.reshape([| batchSize; sequenceLength; heads; headSize |]).transpose (1L, 2L)

        let query =
            attention.Query.forward input
            |> toHeads config.NumAttentionHeads

        let key =
            attention.Key.forward input
            |> toHeads config.NumKeyValueHeads

        let value =
            attention.Value.forward input
            |> toHeads config.NumKeyValueHeads

        let query = applyRotaryEmbedding query cosines sines
        let key = applyRotaryEmbedding key cosines sines

        let key, value =
            match cache with
            | Some(cache: SmolLm2Cache) -> cache.Append(layerIndex, cacheStart, key, value)
            | None -> key, value

        let groups = config.NumAttentionHeads / config.NumKeyValueHeads
        let key = key.repeat_interleave (groups, dim = 1L)
        let value = value.repeat_interleave (groups, dim = 1L)
        let mask = mask |> Option.defaultValue null

        let attended =
            torch.nn.functional.scaled_dot_product_attention (query, key, value, attn_mask = mask, is_casual = isCausal)

        attended.transpose(1L, 2L).contiguous().reshape ([| batchSize; sequenceLength; config.HiddenSize |])
        |> attention.Output.forward

    let private blockForward config layerIndex block input cosines sines mask isCausal cache cacheStart =
        let attended =
            block.InputNorm.forward input
            |> fun hidden ->
                attentionForward config layerIndex block.Attention hidden cosines sines mask isCausal cache cacheStart

        let hidden = input + attended
        let normalized = block.PostAttentionNorm.forward hidden
        let projected = normalized |> block.Mlp.Gate.forward |> _.silu()
        let up = block.Mlp.Up.forward normalized
        let projected = block.Mlp.Down.forward (projected * up)
        hidden + projected

    /// Run SmolLM2. When a cache is supplied, only new tokens should be passed after prefill.
    let forward (input: SmolLm2Input) (model: SmolLm2) : SmolLm2Output =
        if input.InputIds.dtype <> torch.int64 then
            invalidArg (nameof input.InputIds) "SmolLM2 input IDs must use int64 dtype."

        if input.InputIds.shape.Length <> 2 then
            invalidArg (nameof input.InputIds) "SmolLM2 input IDs must have shape [batch, sequence]."

        let batchSize = input.InputIds.shape[0]
        let sequenceLength = input.InputIds.shape[1]

        if batchSize <= 0L || sequenceLength <= 0L then
            invalidArg (nameof input.InputIds) "SmolLM2 input dimensions must be positive."

        let cacheStart = input.Cache |> Option.map _.Length |> Option.defaultValue 0L

        input.Cache
        |> Option.iter (fun cache -> cache.Validate(batchSize, sequenceLength))

        if cacheStart + sequenceLength > model.Config.MaxPositionEmbeddings then
            invalidArg (nameof input.InputIds) $"SmolLM2 accepts at most {model.Config.MaxPositionEmbeddings} positions."

        let positionIds =
            match input.PositionIds with
            | None ->
                torch.arange (cacheStart, cacheStart + sequenceLength, dtype = torch.int64, device = input.InputIds.device)
            | Some positions ->
                if positions.dtype <> torch.int64 then
                    invalidArg (nameof input.PositionIds) "SmolLM2 position IDs must use int64 dtype."

                let validShape =
                    positions.shape = [| sequenceLength |]
                    || positions.shape = [| batchSize; sequenceLength |]

                if not validShape then
                    invalidArg
                        (nameof input.PositionIds)
                        "SmolLM2 position IDs must have shape [sequence] or [batch, sequence]."

                positions

        let totalLength = cacheStart + sequenceLength

        let mask, isCausal =
            attentionMask input batchSize sequenceLength cacheStart totalLength

        let dtype = model.TokenEmbedding.Embeddings.dtype
        let device = input.InputIds.device
        let cosines, sines = rotaryEmbedding model.Config positionIds dtype device
        let hidden = model.TokenEmbedding.forward input.InputIds

        let hidden =
            (hidden, List.indexed model.Blocks)
            ||> List.fold (fun state (index, layer) ->
                blockForward model.Config index layer state cosines sines mask isCausal input.Cache cacheStart)

        let hidden = model.FinalNorm.forward hidden
        let logits = hidden.matmul (model.TokenEmbedding.Embeddings.t ())

        input.Cache
        |> Option.iter (fun cache -> cache.Advance sequenceLength)

        { Logits = logits; Cache = input.Cache }

    /// Load config and a strict single-file or sharded SafeTensors state from a local directory.
    let loadFromDirectory (directory: string) (device: torch.Device) : SmolLm2 * LoadReport =
        if String.IsNullOrWhiteSpace directory then
            invalidArg (nameof directory) "Model directory must not be empty."

        let directory = Path.GetFullPath directory

        if not (Directory.Exists directory) then
            invalidArg (nameof directory) $"Model directory does not exist: '{directory}'."

        let configPath = Path.Combine(directory, "config.json")

        if not (File.Exists configPath) then
            invalidOp $"SmolLM2 config is missing: '{configPath}'."

        let singlePath = Path.Combine(directory, "model.safetensors")
        let indexPath = Path.Combine(directory, "model.safetensors.index.json")

        let reader =
            match File.Exists singlePath, File.Exists indexPath with
            | true, false -> SafeTensors.openFile singlePath
            | false, true -> SafeTensors.openIndex indexPath
            | true, true -> invalidOp "Model directory contains both single-file and sharded SafeTensors state."
            | false, false -> invalidOp "Model directory contains neither model.safetensors nor model.safetensors.index.json."

        use reader = reader
        let config = ConfigJson.load configPath

        let dtype =
            match Map.tryFind "model.embed_tokens.weight" reader.Metadata with
            | Some metadata -> metadata.DType
            | None -> invalidOp "SmolLM2 state is missing 'model.embed_tokens.weight'."

        let model = create config dtype device

        try
            let report = ModelState.loadSafeTensors Strict (state model) reader
            model, report
        with _ ->
            dispose model
            reraise ()
