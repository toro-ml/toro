namespace Toro.Models

open System.IO
open System.Text.Json

module internal DistilGpt2Checkpoint =

    let private label = "DistilGPT-2"
    let private tryProperty root name = JsonConfig.tryProperty root name

    let private int64Element name value =
        JsonConfig.int64Element label name value

    let private int64Value root name = JsonConfig.int64Value label root name
    let private floatValue root name = JsonConfig.floatValue label root name
    let private stringValue root name = JsonConfig.stringValue label root name

    /// Load and validate a Hugging Face config.json file.
    let loadConfig (path: string) =
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
