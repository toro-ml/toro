# Toro.Models.DistilGpt2

`Toro.Models.DistilGpt2` provides the DistilGPT-2 architecture, Hugging Face-compatible named state, local SafeTensors loading, GPT-2 Conv1D projections, and a fixed-capacity KV cache.

The public API remains in the `Toro.Models` namespace. Use `DistilGpt2.asCausalLm` to bind a loaded model to the shared generation API from `Toro.Models`.
