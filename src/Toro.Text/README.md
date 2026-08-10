# Toro.Text

[![NuGet](https://img.shields.io/nuget/v/Toro.Text.svg)](https://www.nuget.org/packages/Toro.Text)

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

let r = result {
    let! tensor = Encode.toTensor tokenizer "hello world" 16 0 Cpu
}
```

## Features

- **Tokenizer wrappers** -- Tiktoken, BPE, WordPiece, SentencePiece via config records
- **Encode** -- convert text to Toro tensors for model input

## License

[MIT](https://github.com/toro-ml/toro/blob/main/LICENSE)
