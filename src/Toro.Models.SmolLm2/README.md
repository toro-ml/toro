# Toro.Models.SmolLm2

`Toro.Models.SmolLm2` provides the SmolLM2 architecture, Hugging Face-compatible named state, local SafeTensors loading, grouped-query attention, rotary position embeddings, and a fixed-capacity KV cache.

The public API remains in the `Toro.Models` namespace. Use `SmolLm2.asCausalLm` to bind a loaded model to the shared generation API from `Toro.Models`.
