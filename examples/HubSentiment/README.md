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
2. **Tokenize** — `Tokenizer.fromBert` wraps Microsoft.ML.Tokenizers `BertTokenizer`, including CJK splitting and `[CLS]`/`[SEP]`.
3. **Name mapping** — `NameMapping` translates HF-style safetensors keys (e.g. `distilbert.transformer.layer.0.attention.q_lin.weight`) to Toro parameter paths (e.g. `Layers.0.Attn.WQ.Weight`).
4. **Model** — Layers are `PostNormTransformerBlock` (Attn → Add → Norm). Embeddings and the classification head stay as a small DistilBERT-specific record.
5. **Inference** — Each sentence runs through embeddings, 6 transformer layers, and a classification head to produce POSITIVE/NEGATIVE labels.
