# TextClassifier

Char-level sentiment classification using a `PreNormTransformerBlock`.

## Run

```bash
dotnet run --project examples/TextClassifier
```

## Concepts

- `Embedding` + `PreNormTransformerBlock` + mean pooling + `Linear` head
- `MultiHeadAttention` with scaled dot-product attention
- `Loss.crossEntropy` for multi-class classification
- Inference on unseen words with `Toro.noGrad`
