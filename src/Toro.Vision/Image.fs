namespace Toro.Vision

open System.IO
open System.Runtime.InteropServices
open SkiaSharp
open TorchSharp
open Toro

/// Supported image output formats.
type ImageFormat =
    | Jpeg
    | Png
    | Webp

/// Image I/O and SKBitmap-Tensor conversion.
module Image =

    let private ensureImager () =
        if isNull (box TorchSharp.torchvision.io.DefaultImager) then
            TorchSharp.torchvision.io.DefaultImager <- TorchSharp.torchvision.io.SkiaImager()

    let private toTorchvisionFormat =
        function
        | Jpeg -> TorchSharp.torchvision.ImageFormat.Jpeg
        | Png -> TorchSharp.torchvision.ImageFormat.Png
        | Webp -> TorchSharp.torchvision.ImageFormat.Png

    let private toSkFormat =
        function
        | Jpeg -> SKEncodedImageFormat.Jpeg
        | Png -> SKEncodedImageFormat.Png
        | Webp -> SKEncodedImageFormat.Webp

    /// Convert an SKBitmap to a [3, H, W] float32 tensor in [0, 1].
    /// Alpha channel is discarded; output has 3 channels (RGB).
    let toTensor (bitmap: SKBitmap) (device: Device) : Result<Tensor, ToroError> =
        let w = bitmap.Width
        let h = bitmap.Height

        let rgba, shouldDispose =
            if bitmap.ColorType = SKColorType.Rgba8888 then
                bitmap, false
            else
                let converted = new SKBitmap(w, h, SKColorType.Rgba8888, SKAlphaType.Unpremul)
                bitmap.CopyTo(converted, SKColorType.Rgba8888) |> ignore
                converted, true

        let pixels = rgba.GetPixels()
        let data = Array.zeroCreate<float32> (3 * h * w)

        for y in 0 .. h - 1 do
            for x in 0 .. w - 1 do
                let offset = nativeint ((y * w + x) * 4)
                let r = Marshal.ReadByte(pixels + offset)
                let g = Marshal.ReadByte(pixels + offset + 1n)
                let b = Marshal.ReadByte(pixels + offset + 2n)
                data[0 * h * w + y * w + x] <- float32 r / 255.0f
                data[1 * h * w + y * w + x] <- float32 g / 255.0f
                data[2 * h * w + y * w + x] <- float32 b / 255.0f

        if shouldDispose then
            rgba.Dispose()

        Tensor.ofArray (data, device)
        |> Result.bind (fun t -> t.reshape [ 3; h; w ])

    /// Convert a [3, H, W] float32 tensor in [0, 1] to an SKBitmap.
    let fromTensor (tensor: Tensor) : Result<SKBitmap, ToroError> =
        ToroError.wrap (fun () ->
            let t = tensor.Inner.cpu().contiguous ()
            let shape = t.shape

            if shape.Length <> 3 || shape[0] <> 3L then
                failwith $"fromTensor: expected shape [3, H, W], got {shape}"

            let h = int shape[1]
            let w = int shape[2]
            let bitmap = new SKBitmap(w, h, SKColorType.Rgba8888, SKAlphaType.Unpremul)
            let pixels = bitmap.GetPixels()
            let d = t.data<float32> ()

            for y in 0 .. h - 1 do
                for x in 0 .. w - 1 do
                    let ri = float32 (d[0 * h * w + y * w + x]) * 255.0f
                    let gi = float32 (d[1 * h * w + y * w + x]) * 255.0f
                    let bi = float32 (d[2 * h * w + y * w + x]) * 255.0f
                    let r = byte (System.Math.Clamp(ri, 0.0f, 255.0f))
                    let g = byte (System.Math.Clamp(gi, 0.0f, 255.0f))
                    let b = byte (System.Math.Clamp(bi, 0.0f, 255.0f))
                    let offset = nativeint ((y * w + x) * 4)
                    Marshal.WriteByte(pixels + offset, r)
                    Marshal.WriteByte(pixels + offset + 1n, g)
                    Marshal.WriteByte(pixels + offset + 2n, b)
                    Marshal.WriteByte(pixels + offset + 3n, 255uy)

            bitmap)

    /// Load an image file as a [3, H, W] float32 tensor in [0, 1].
    let load (path: string) (device: Device) : Result<Tensor, ToroError> =
        ToroError.wrap (fun () ->
            ensureImager ()
            let t = TorchSharp.torchvision.io.read_image (path, TorchSharp.torchvision.io.ImageReadMode.RGB)
            t.to_type(torch.float32).div(torch.tensor 255.0f).``to`` (Device.toTorch device))
        |> Result.bind Tensor.ofTorchTensor

    /// Load an image from a stream as a [3, H, W] float32 tensor in [0, 1].
    let loadStream (stream: Stream) (device: Device) : Result<Tensor, ToroError> =
        ToroError.wrap (fun () ->
            ensureImager ()
            use ms = new MemoryStream()
            stream.CopyTo(ms)
            let bytes = ms.ToArray()
            let bmp = SKBitmap.Decode(bytes)
            bmp)
        |> Result.bind (fun bmp ->
            let r = toTensor bmp device
            bmp.Dispose()
            r)

    /// Save a [3, H, W] float32 tensor as an image file.
    let save (tensor: Tensor) (path: string) (format: ImageFormat) (_quality: int) : Result<unit, ToroError> =
        ToroError.wrap (fun () ->
            ensureImager ()
            let t = (tensor.Inner * torch.tensor 255.0f).clamp (torch.tensor 0.0f, torch.tensor 255.0f)
            let t = t.to_type(torch.uint8)
            TorchSharp.torchvision.io.write_image (t, path, toTorchvisionFormat format))

    /// Save a batch tensor [N, C, H, W] as a grid image with nrow images per row.
    let saveGrid
        (tensor: Tensor)
        (path: string)
        (format: ImageFormat)
        (quality: int)
        (nrow: int)
        : Result<unit, ToroError> =
        ToroError.wrap (fun () ->
            ensureImager ()
            TorchSharp.torchvision.utils.save_image (
                tensor.Inner,
                path,
                toTorchvisionFormat format,
                nrow = int64 nrow
            ))
