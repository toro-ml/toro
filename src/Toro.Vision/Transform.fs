namespace Toro.Vision

open Toro

/// Image transform that maps a tensor to a tensor.
type ITransform =
    abstract apply: Tensor -> Result<Tensor, ToroError>

/// Compose multiple transforms into a single pipeline.
module Compose =
    /// Apply transforms sequentially, short-circuiting on the first error.
    let apply (transforms: ITransform list) (x: Tensor) : Result<Tensor, ToroError> =
        let rec loop (ts: ITransform list) (t: Tensor) =
            match ts with
            | [] -> Ok t
            | h :: rest ->
                match h.apply t with
                | Ok t' -> loop rest t'
                | err -> err

        loop transforms x

/// Channel-wise normalization: $(x - \mu) / \sigma$.
/// Input: $[C, H, W]$ or $[B, C, H, W]$.
type Normalize = {
    Mean: float list
    Std: float list
} with

    member this.apply(x: Tensor) : Result<Tensor, ToroError> =
        let rank = x.Rank
        let cDim = if rank = 4 then 1 else 0
        let channels = x.Shape[cDim]

        if channels <> this.Mean.Length || channels <> this.Std.Length then
            Error(Msg $"Normalize: expected {channels} channels, got mean={this.Mean.Length}, std={this.Std.Length}")
        else
            result {
                let shape =
                    if rank = 4 then
                        [ 1; channels; 1; 1 ]
                    else
                        [ channels; 1; 1 ]

                let! meanT =
                    Tensor.ofArray (this.Mean |> List.map float32 |> List.toArray, x.Device)
                    |> Result.bind (fun t -> t.reshape shape)

                let! stdT =
                    Tensor.ofArray (this.Std |> List.map float32 |> List.toArray, x.Device)
                    |> Result.bind (fun t -> t.reshape shape)

                return (x - meanT) / stdT
            }

    interface ITransform with
        member this.apply x = this.apply x

module Normalize =
    /// ImageNet normalization preset.
    let imageNet: Normalize = {
        Mean = [ 0.485; 0.456; 0.406 ]
        Std = [ 0.229; 0.224; 0.225 ]
    }

/// Resize to the target spatial size using bilinear interpolation.
/// Input: $[C, H, W]$ or $[B, C, H, W]$.
type Resize = {
    Height: int
    Width: int
} with

    member this.apply(x: Tensor) : Result<Tensor, ToroError> =
        let rank = x.Rank

        result {
            if rank = 3 then
                let! batched = x.unsqueeze 0
                let! resized = batched.interpolate ([ this.Height; this.Width ], Bilinear)
                return! resized.squeeze 0
            else
                return! x.interpolate ([ this.Height; this.Width ], Bilinear)
        }

    interface ITransform with
        member this.apply x = this.apply x

module Resize =
    let create (height: int) (width: int) : Resize = { Height = height; Width = width }

/// Randomly flip the image horizontally with probability p.
/// Input: $[C, H, W]$ or $[B, C, H, W]$.
type RandomHorizontalFlip = {
    P: float
} with

    member this.apply(x: Tensor) : Result<Tensor, ToroError> =
        let wDim = x.Rank - 1

        if System.Random.Shared.NextDouble() < this.P then
            x.flip [ wDim ]
        else
            Ok x

    interface ITransform with
        member this.apply x = this.apply x

module RandomHorizontalFlip =
    let create (p: float) : RandomHorizontalFlip = { P = p }
    let defaultFlip: RandomHorizontalFlip = { P = 0.5 }

/// Randomly crop the image to the given size.
/// Input: $[C, H, W]$ or $[B, C, H, W]$.
type RandomCrop = {
    Height: int
    Width: int
} with

    member this.apply(x: Tensor) : Result<Tensor, ToroError> =
        let rank = x.Rank
        let hDim = rank - 2
        let wDim = rank - 1
        let h = x.Shape[hDim]
        let w = x.Shape[wDim]

        if h < this.Height || w < this.Width then
            Error(Msg $"RandomCrop: input {h}x{w} is smaller than crop {this.Height}x{this.Width}")
        else
            let top = System.Random.Shared.Next(0, h - this.Height + 1)
            let left = System.Random.Shared.Next(0, w - this.Width + 1)

            result {
                let! cropped = x.narrow (hDim, int64 top, int64 this.Height)
                return! cropped.narrow (wDim, int64 left, int64 this.Width)
            }

    interface ITransform with
        member this.apply x = this.apply x

module RandomCrop =
    let create (height: int) (width: int) : RandomCrop = { Height = height; Width = width }

/// Crop the center region of the image.
/// Input: $[C, H, W]$ or $[B, C, H, W]$.
type CenterCrop = {
    Height: int
    Width: int
} with

    member this.apply(x: Tensor) : Result<Tensor, ToroError> =
        let rank = x.Rank
        let hDim = rank - 2
        let wDim = rank - 1
        let h = x.Shape[hDim]
        let w = x.Shape[wDim]

        if h < this.Height || w < this.Width then
            Error(Msg $"CenterCrop: input {h}x{w} is smaller than crop {this.Height}x{this.Width}")
        else
            let top = (h - this.Height) / 2
            let left = (w - this.Width) / 2

            result {
                let! cropped = x.narrow (hDim, int64 top, int64 this.Height)
                return! cropped.narrow (wDim, int64 left, int64 this.Width)
            }

    interface ITransform with
        member this.apply x = this.apply x

module CenterCrop =
    let create (height: int) (width: int) : CenterCrop = { Height = height; Width = width }

/// Randomly flip the image vertically with probability p.
/// Input: $[C, H, W]$ or $[B, C, H, W]$.
type RandomVerticalFlip = {
    P: float
} with

    member this.apply(x: Tensor) : Result<Tensor, ToroError> =
        let hDim = x.Rank - 2

        if System.Random.Shared.NextDouble() < this.P then
            x.flip [ hDim ]
        else
            Ok x

    interface ITransform with
        member this.apply x = this.apply x

module RandomVerticalFlip =
    let create (p: float) : RandomVerticalFlip = { P = p }
    let defaultFlip: RandomVerticalFlip = { P = 0.5 }

/// Convert an RGB image to grayscale using ITU-R BT.601 weights.
/// Input: $[3, H, W]$ or $[B, 3, H, W]$.
type ToGrayscale = {
    NumOutputChannels: int
} with

    member this.apply(x: Tensor) : Result<Tensor, ToroError> =
        let rank = x.Rank
        let cDim = if rank = 4 then 1 else 0
        let channels = x.Shape[cDim]

        if channels <> 3 then
            Error(Msg $"ToGrayscale: expected 3 channels, got {channels}")
        else
            result {
                let! r = x.narrow (cDim, 0L, 1L)
                let! g = x.narrow (cDim, 1L, 1L)
                let! b = x.narrow (cDim, 2L, 1L)

                let gray = 0.2989 * r + 0.587 * g + 0.114 * b

                if this.NumOutputChannels = 1 then
                    return gray
                else
                    return! Tensor.cat ([ gray; gray; gray ], cDim)
            }

    interface ITransform with
        member this.apply x = this.apply x

module ToGrayscale =
    let create (numOutputChannels: int) : ToGrayscale = {
        NumOutputChannels = numOutputChannels
    }

    let single: ToGrayscale = { NumOutputChannels = 1 }
    let triple: ToGrayscale = { NumOutputChannels = 3 }

/// Convert image tensor dtype, scaling between [0, 255] int and [0.0, 1.0] float ranges.
/// Input: any image tensor.
type ConvertImageDType = {
    TargetDType: DType
} with

    member this.apply(x: Tensor) : Result<Tensor, ToroError> =
        let isFloatDType dt =
            match dt with
            | F16
            | BF16
            | F32
            | F64 -> true
            | _ -> false

        let srcFloat = isFloatDType x.DType
        let dstFloat = isFloatDType this.TargetDType

        result {
            if srcFloat && not dstFloat then
                let scaled = x * 255.0
                return! scaled.toDType this.TargetDType
            elif (not srcFloat) && dstFloat then
                let! converted = x.toDType this.TargetDType
                return converted / 255.0
            else
                return! x.toDType this.TargetDType
        }

    interface ITransform with
        member this.apply x = this.apply x

module ConvertImageDType =
    let create (dtype: DType) : ConvertImageDType = { TargetDType = dtype }
