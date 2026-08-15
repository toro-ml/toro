# HubSentiment

Load a pre-trained [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) sentiment classifier from Hugging Face Hub and run inference.

This example shows how to:

- Download safetensors weights with `Hub.download`
- Map HF-style parameter names to Toro record field paths
- Load weights into a custom model with `ModelState.loadSafeTensorsWith`

## Run

```bash
dotnet run --project examples/HubSentiment
```

## Output

```
Downloading distilbert/distilbert-base-uncased-finetuned-sst-2-english ...
Downloaded 104 tensors, vocab 30522 tokens
Model loaded

  this movie is great       -> POSITIVE
  i love it so much         -> POSITIVE
  what a wonderful day      -> POSITIVE
  this is terrible          -> NEGATIVE
  worst film ever made       -> NEGATIVE
  i hate this movie         -> NEGATIVE
```

## How it works

1. **Download** — `Hub.download` fetches `model.safetensors` (268 MB) and `vocab.txt` from HF Hub. Both files are cached in `~/.cache/toro/hub/`.
2. **Tokenize** — A greedy longest-match WordPiece tokenizer splits input text into subword token IDs using the downloaded vocabulary.
3. **Name mapping** — `buildNameMap` translates 104 HF-style safetensors keys (e.g. `distilbert.transformer.layer.0.attention.q_lin.weight`) to Toro parameter paths (e.g. `Layers.0.Attention.Q.Weight`).
4. **Model** — DistilBERT uses post-norm (Attn → Add → Norm), which differs from `Toro.NN.TransformerBlock` (pre-norm). Custom record types (`DistilBertAttention`, `DistilBertLayer`, `DistilBertClassifier`) implement the exact architecture.
5. **Inference** — Each sentence runs through embeddings, 6 transformer layers, and a classification head to produce POSITIVE/NEGATIVE labels.
