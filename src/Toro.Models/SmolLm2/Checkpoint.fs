namespace Toro.Models

open System.IO
open System.Text.Json

module internal SmolLm2Checkpoint =

    let private label = "SmolLM2"
    let private property root name = JsonConfig.property label root name
    let private int64Value root name = JsonConfig.int64Value label root name
    let private floatValue root name = JsonConfig.floatValue label root name
    let private boolValue root name = JsonConfig.boolValue label root name
    let private stringValue root name = JsonConfig.stringValue label root name

    /// Load and validate a Hugging Face config.json file.
    let loadConfig (path: string) =
        use document = JsonDocument.Parse(File.ReadAllText path)
        let root = document.RootElement

        JsonConfig.validateObject label root

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
