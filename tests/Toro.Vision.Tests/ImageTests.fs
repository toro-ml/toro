module ImageTests

open Xunit
open FsUnit.Xunit
open SkiaSharp
open Toro
open Toro.Vision
open TestHelper

[<Fact>]
let ``toTensor produces [3, H, W] from SKBitmap`` () =
    let bmp = new SKBitmap(4, 3, SKColorType.Rgba8888, SKAlphaType.Unpremul)
    let pixels = bmp.GetPixels()

    for y in 0..2 do
        for x in 0..3 do
            let offset = nativeint ((y * 4 + x) * 4)
            System.Runtime.InteropServices.Marshal.WriteByte(pixels + offset, 255uy)
            System.Runtime.InteropServices.Marshal.WriteByte(pixels + offset + 1n, 128uy)
            System.Runtime.InteropServices.Marshal.WriteByte(pixels + offset + 2n, 0uy)
            System.Runtime.InteropServices.Marshal.WriteByte(pixels + offset + 3n, 255uy)

    let t = Image.toTensor bmp Cpu
    t.Shape |> should equal [ 3; 3; 4 ]

    t.at [ I 0; I 0; I 0 ]
    |> scalarF32
    |> should (equalWithin 1e-2) 1.0f

    t.at [ I 1; I 0; I 0 ]
    |> scalarF32
    |> should (equalWithin 1e-2) (128.0f / 255.0f)

    t.at [ I 2; I 0; I 0 ]
    |> scalarF32
    |> should (equalWithin 1e-2) 0.0f

    bmp.Dispose()

[<Fact>]
let ``fromTensor roundtrips correctly`` () =
    let bmp = new SKBitmap(2, 2, SKColorType.Rgba8888, SKAlphaType.Unpremul)
    let pixels = bmp.GetPixels()

    for i in 0..3 do
        let offset = nativeint (i * 4)
        System.Runtime.InteropServices.Marshal.WriteByte(pixels + offset, 200uy)
        System.Runtime.InteropServices.Marshal.WriteByte(pixels + offset + 1n, 100uy)
        System.Runtime.InteropServices.Marshal.WriteByte(pixels + offset + 2n, 50uy)
        System.Runtime.InteropServices.Marshal.WriteByte(pixels + offset + 3n, 255uy)

    let t = Image.toTensor bmp Cpu
    let bmp2 = Image.fromTensor t

    bmp2.Width |> should equal 2
    bmp2.Height |> should equal 2

    let p = bmp2.GetPixels()

    System.Runtime.InteropServices.Marshal.ReadByte(p)
    |> should equal 200uy

    System.Runtime.InteropServices.Marshal.ReadByte(p + 1n)
    |> should equal 100uy

    System.Runtime.InteropServices.Marshal.ReadByte(p + 2n)
    |> should equal 50uy

    bmp.Dispose()
    bmp2.Dispose()

[<Fact>]
let ``SkiaTransform resize produces correct dimensions`` () =
    let bmp = new SKBitmap(100, 80, SKColorType.Rgba8888, SKAlphaType.Unpremul)
    let resized = SkiaTransform.resize 50 40 bmp
    resized.Width |> should equal 50
    resized.Height |> should equal 40
    bmp.Dispose()
    resized.Dispose()

[<Fact>]
let ``SkiaTransform centerCrop produces correct dimensions`` () =
    let bmp = new SKBitmap(100, 80, SKColorType.Rgba8888, SKAlphaType.Unpremul)
    let cropped = SkiaTransform.centerCrop 50 40 bmp
    cropped.Width |> should equal 50
    cropped.Height |> should equal 40
    bmp.Dispose()
    cropped.Dispose()

[<Fact>]
let ``SkiaTransform centerCrop rejects too-small input`` () =
    let bmp = new SKBitmap(30, 30, SKColorType.Rgba8888, SKAlphaType.Unpremul)

    try
        SkiaTransform.centerCrop 50 40 bmp |> ignore
        failwith "Expected exception"
    with _ ->
        ()

    bmp.Dispose()

[<Fact>]
let ``SkiaTransform flipH preserves dimensions`` () =
    let bmp = new SKBitmap(10, 8, SKColorType.Rgba8888, SKAlphaType.Unpremul)
    let flipped = SkiaTransform.flipH bmp
    flipped.Width |> should equal 10
    flipped.Height |> should equal 8
    bmp.Dispose()
    flipped.Dispose()

[<Fact>]
let ``SkiaTransform pipeline composes and produces tensor`` () =
    let bmp = new SKBitmap(100, 100, SKColorType.Rgba8888, SKAlphaType.Unpremul)

    let result =
        SkiaTransform.pipeline [ SkiaTransform.resize 50 50; SkiaTransform.flipH ] bmp Cpu

    let t = result
    t.Shape |> should equal [ 3; 50; 50 ]
    bmp.Dispose()

[<Fact>]
let ``Normalize direct member call works without cast`` () =
    let x = Tensor.ones ([ 3; 4; 4 ], F32, Cpu)
    let norm = Normalize.imageNet
    let out = norm.apply x
    out.Shape |> should equal [ 3; 4; 4 ]

[<Fact>]
let ``Resize direct member call works without cast`` () =
    let x = Tensor.randn ([ 3; 32; 32 ], F32, Cpu)
    let r = Resize.create 16 16
    let out = r.apply x
    out.Shape |> should equal [ 3; 16; 16 ]
