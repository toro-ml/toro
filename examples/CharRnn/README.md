# CharRnn

Character-level text generation with LSTM, trained on a Shakespeare excerpt.

## Run

```bash
dotnet run --project examples/CharRnn
```

## Concepts

- `Embedding` + `LSTM` + `Linear` record model
- `LSTM.seq` for sequence processing, `LSTM.step` for autoregressive generation
- `torch.multinomial` for sampling from predicted probabilities
