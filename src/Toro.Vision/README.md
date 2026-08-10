# Toro.Vision

[![NuGet](https://img.shields.io/nuget/v/Toro.Vision.svg)](https://www.nuget.org/packages/Toro.Vision)

Image I/O and transforms for [Toro](https://www.nuget.org/packages/Toro). TorchVision-backed loading and saving, SkiaSharp bitmap preprocessing, and composable tensor transforms.

**[Documentation](https://toro-ml.github.io/toro/)**

## Installation

```bash
dotnet add package Toro.Vision
dotnet add package TorchSharp-cpu
```

## Quick Example

```fsharp
open Toro
open Toro.Vision

let r = result {
    let! img = Image.load "photo.jpg" Cpu
    let! resized = (Resize.create 224 224).apply img
    let! normalized = Normalize.imageNet.apply resized
    do! Image.save normalized "out.jpg" Jpeg 0
}
```

## Features

- **Image I/O** -- load and save via TorchVision; SKBitmap ↔ Tensor conversion
- **SkiaTransform** -- CPU-side resize, crop, and flip on bitmaps before tensor conversion
- **ITransform** -- composable tensor transforms (`Resize`, `Normalize`, flips, crops, and related ops)

## License

[MIT](https://github.com/toro-ml/toro/blob/main/LICENSE)
