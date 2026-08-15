# Toro.Extensions.AI

`Toro.Extensions.AI` adapts a `Toro.Models.CausalLm` to `Microsoft.Extensions.AI.IChatClient`.

The caller supplies model-specific prompt formatting and token encode/decode functions. The adapter supports text messages, maximum output length, temperature sampling, streaming, and cancellation. Tool calls and non-text content are rejected.
