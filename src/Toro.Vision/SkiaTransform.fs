namespace Toro.Vision

open SkiaSharp
open Toro

/// Spatial transforms operating directly on SKBitmap for CPU-optimized preprocessing.
module SkiaTransform =

    /// Resize a bitmap to the given width and height using high-quality sampling.
    let resize (width: int) (height: int) (bitmap: SKBitmap) : SKBitmap =
        let resized = new SKBitmap(width, height, bitmap.ColorType, bitmap.AlphaType)
        let sampling = SKSamplingOptions(SKCubicResampler.Mitchell)
        bitmap.ScalePixels(resized, sampling) |> ignore
        resized

    /// Center crop a bitmap to the given width and height.
    let centerCrop (width: int) (height: int) (bitmap: SKBitmap) : SKBitmap =
        if bitmap.Width < width || bitmap.Height < height then
            failwith $"centerCrop: input {bitmap.Width}x{bitmap.Height} is smaller than crop {width}x{height}"
        else
            let left = (bitmap.Width - width) / 2
            let top = (bitmap.Height - height) / 2
            let rect = SKRectI(left, top, left + width, top + height)
            let cropped = new SKBitmap(width, height, bitmap.ColorType, bitmap.AlphaType)
            bitmap.ExtractSubset(cropped, rect) |> ignore
            cropped

    /// Random crop a bitmap to the given width and height.
    let randomCrop (width: int) (height: int) (bitmap: SKBitmap) : SKBitmap =
        if bitmap.Width < width || bitmap.Height < height then
            failwith $"randomCrop: input {bitmap.Width}x{bitmap.Height} is smaller than crop {width}x{height}"
        else
            let left = System.Random.Shared.Next(0, bitmap.Width - width + 1)
            let top = System.Random.Shared.Next(0, bitmap.Height - height + 1)
            let rect = SKRectI(left, top, left + width, top + height)
            let cropped = new SKBitmap(width, height, bitmap.ColorType, bitmap.AlphaType)
            bitmap.ExtractSubset(cropped, rect) |> ignore
            cropped

    /// Flip a bitmap horizontally.
    let flipH (bitmap: SKBitmap) : SKBitmap =
        let w = bitmap.Width
        let h = bitmap.Height
        let flipped = new SKBitmap(w, h, bitmap.ColorType, bitmap.AlphaType)
        use canvas = new SKCanvas(flipped)
        canvas.Scale(-1.0f, 1.0f)
        canvas.Translate(float32 -w, 0.0f)
        canvas.DrawBitmap(bitmap, 0.0f, 0.0f)
        flipped

    /// Flip a bitmap vertically.
    let flipV (bitmap: SKBitmap) : SKBitmap =
        let w = bitmap.Width
        let h = bitmap.Height
        let flipped = new SKBitmap(w, h, bitmap.ColorType, bitmap.AlphaType)
        use canvas = new SKCanvas(flipped)
        canvas.Scale(1.0f, -1.0f)
        canvas.Translate(0.0f, float32 -h)
        canvas.DrawBitmap(bitmap, 0.0f, 0.0f)
        flipped

    /// Apply a list of bitmap transforms as a pipeline, then convert to a tensor.
    let pipeline (transforms: (SKBitmap -> SKBitmap) list) (bitmap: SKBitmap) (device: Device) : Tensor =
        let mutable current = bitmap

        for t in transforms do
            let next = t current

            if not (obj.ReferenceEquals(current, bitmap)) then
                current.Dispose()

            current <- next

        let result = Image.toTensor current device

        if not (obj.ReferenceEquals(current, bitmap)) then
            current.Dispose()

        result
