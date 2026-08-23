# Toro.Text

[![Toro.Text](https://img.shields.io/nuget/v/Toro.Text.svg?label=Toro.Text)](https://www.nuget.org/packages/Toro.Text)

Text tokenization bridge between [Microsoft.ML.Tokenizers](https://www.nuget.org/packages/Microsoft.ML.Tokenizers) and [Toro](https://www.nuget.org/packages/Toro) tensors.

**[Documentation](https://toro-ml.github.io/toro/)**

## Installation

```bash
dotnet add package Toro.Text
dotnet add package TorchSharp-cpu
```

## Quick Example

```fsharp
open Toro
open Toro.Text

let tokenizer = Tokenizer.fromTiktoken (TiktokenConfig.create "gpt-4o")

let ids = tokenizer.encode "hello world"
let text = tokenizer.decode ids

let tensor = Encode.toTensor tokenizer "hello world" 16 0 Cpu
```

## Features

- **Tokenizer wrappers** -- Tiktoken, BPE, WordPiece, SentencePiece, BERT via config records
- **Encode** -- convert text to Toro tensors for model input (`Collation`), with fixed or batch-max padding

## License

[MIT](https://github.com/toro-ml/toro/blob/main/LICENSE)
