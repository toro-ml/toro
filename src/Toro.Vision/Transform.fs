namespace Toro.Vision

open TorchSharp
open Toro

module private TorchRandom =
    let nextDouble () =
        use value = torch.rand ([| 1L |], dtype = torch.float64, device = torch.CPU)
        value.item<double> ()

    let nextInt (exclusiveUpperBound: int) =
        use value =
            torch.randint (int64 exclusiveUpperBound, [| 1L |], dtype = torch.int64, device = torch.CPU)

        value.item<int64> () |> int

/// Image transform that maps a tensor to a tensor.
type ITransform =
    abstract apply: Tensor -> Tensor

/// Compose multiple transforms into a single pipeline.
module Compose =
    /// Apply transforms sequentially.
    let apply (transforms: ITransform list) (x: Tensor) : Tensor =
        let rec loop (ts: ITransform list) (t: Tensor) =
            match ts with
            | [] -> t
            | h :: rest -> loop rest (h.apply t)

        loop transforms x

/// Channel-wise normalization: $(x - \mu) / \sigma$.
/// Input: $[C, H, W]$ or $[B, C, H, W]$.
type Normalize = {
    Mean: float list
    Std: float list
} with

    member this.apply(x: Tensor) : Tensor =
        let rank = int x.ndim
        let cDim = if rank = 4 then 1 else 0
        let channels = int x.shape[cDim]

        if channels <> this.Mean.Length || channels <> this.Std.Length then
            failwith $"Normalize: expected {channels} channels, got mean={this.Mean.Length}, std={this.Std.Length}"
        else
            let shape =
                if rank = 4 then
                    [| 1L; int64 channels; 1L; 1L |]
                else
                    [| int64 channels; 1L; 1L |]

            let meanT =
                torch.tensor (this.Mean |> List.map float32 |> List.toArray, device = x.device)
                |> fun t -> t.reshape shape

            let stdT =
                torch.tensor (this.Std |> List.map float32 |> List.toArray, device = x.device)
                |> fun t -> t.reshape shape

            (x - meanT) / stdT

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

    member this.apply(x: Tensor) : Tensor =
        let rank = int x.ndim
        let sz = [| int64 this.Height; int64 this.Width |]

        if rank = 3 then
            let batched = x.unsqueeze 0L

            let resized =
                torch.nn.functional.interpolate (batched, sz, mode = torch.InterpolationMode.Bilinear)

            resized.squeeze 0L
        else
            torch.nn.functional.interpolate (x, sz, mode = torch.InterpolationMode.Bilinear)

    interface ITransform with
        member this.apply x = this.apply x

module Resize =
    let create (height: int) (width: int) : Resize = { Height = height; Width = width }

/// Resize an image while preserving its aspect ratio so that its shortest edge has the requested size.
/// Input: $[C, H, W]$ or $[B, C, H, W]$.
type ResizeShortestEdge = {
    Size: int
    Mode: torch.InterpolationMode
} with

    member this.apply(x: Tensor) : Tensor =
        if this.Size <= 0 then
            invalidArg (nameof this.Size) "ResizeShortestEdge size must be positive."

        let rank = int x.ndim

        if rank <> 3 && rank <> 4 then
            invalidArg (nameof x) $"ResizeShortestEdge expects rank 3 or 4, got rank {rank}."

        let h = int x.shape[rank - 2]
        let w = int x.shape[rank - 1]

        if h <= 0 || w <= 0 then
            invalidArg (nameof x) $"ResizeShortestEdge expects positive spatial dimensions, got {h}x{w}."

        let scale = float this.Size / float (min h w)
        let newHeight = max 1 (int (float h * scale))
        let newWidth = max 1 (int (float w * scale))
        let size = [| int64 newHeight; int64 newWidth |]

        let resize input =
            match this.Mode with
            | torch.InterpolationMode.Bilinear
            | torch.InterpolationMode.Bicubic ->
                torch.nn.functional.interpolate (input, size, mode = this.Mode, align_corners = false)
            | _ -> torch.nn.functional.interpolate (input, size, mode = this.Mode)

        if rank = 3 then
            let batched = x.unsqueeze 0L
            let resized = resize batched
            resized.squeeze 0L
        else
            resize x

    interface ITransform with
        member this.apply x = this.apply x

module ResizeShortestEdge =
    /// Create an aspect-ratio-preserving shortest-edge resize.
    let create (size: int) (mode: torch.InterpolationMode) : ResizeShortestEdge = { Size = size; Mode = mode }

/// Randomly flip the image horizontally with probability p.
/// Input: $[C, H, W]$ or $[B, C, H, W]$.
type RandomHorizontalFlip = {
    P: float
} with

    member this.apply(x: Tensor) : Tensor =
        let wDim = int x.ndim - 1

        if TorchRandom.nextDouble () < this.P then
            x.flip [| int64 wDim |]
        else
            x

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

    member this.apply(x: Tensor) : Tensor =
        let rank = int x.ndim
        let hDim = rank - 2
        let wDim = rank - 1
        let h = int x.shape[hDim]
        let w = int x.shape[wDim]

        if h < this.Height || w < this.Width then
            failwith $"RandomCrop: input {h}x{w} is smaller than crop {this.Height}x{this.Width}"
        else
            let top = TorchRandom.nextInt (h - this.Height + 1)
            let left = TorchRandom.nextInt (w - this.Width + 1)

            let cropped = x.narrow (hDim, int64 top, int64 this.Height)
            cropped.narrow (wDim, int64 left, int64 this.Width)

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

    member this.apply(x: Tensor) : Tensor =
        let rank = int x.ndim
        let hDim = rank - 2
        let wDim = rank - 1
        let h = int x.shape[hDim]
        let w = int x.shape[wDim]

        if h < this.Height || w < this.Width then
            failwith $"CenterCrop: input {h}x{w} is smaller than crop {this.Height}x{this.Width}"
        else
            let top = (h - this.Height) / 2
            let left = (w - this.Width) / 2

            let cropped = x.narrow (hDim, int64 top, int64 this.Height)
            cropped.narrow (wDim, int64 left, int64 this.Width)

    interface ITransform with
        member this.apply x = this.apply x

module CenterCrop =
    let create (height: int) (width: int) : CenterCrop = { Height = height; Width = width }

/// Randomly flip the image vertically with probability p.
/// Input: $[C, H, W]$ or $[B, C, H, W]$.
type RandomVerticalFlip = {
    P: float
} with

    member this.apply(x: Tensor) : Tensor =
        let hDim = int x.ndim - 2

        if TorchRandom.nextDouble () < this.P then
            x.flip [| int64 hDim |]
        else
            x

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

    member this.apply(x: Tensor) : Tensor =
        let rank = int x.ndim
        let cDim = if rank = 4 then 1 else 0
        let channels = int x.shape[cDim]

        if channels <> 3 then
            failwith $"ToGrayscale: expected 3 channels, got {channels}"
        else
            let r = x.narrow (cDim, 0L, 1L)
            let g = x.narrow (cDim, 1L, 1L)
            let b = x.narrow (cDim, 2L, 1L)

            let gray = 0.2989 * r + 0.587 * g + 0.114 * b

            if this.NumOutputChannels = 1 then
                gray
            else
                torch.cat ([| gray; gray; gray |], int64 cDim)

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
    TargetDType: torch.ScalarType
} with

    member this.apply(x: Tensor) : Tensor =
        let isFloatDType (dt: torch.ScalarType) =
            match dt with
            | torch.ScalarType.Float16
            | torch.ScalarType.BFloat16
            | torch.ScalarType.Float32
            | torch.ScalarType.Float64 -> true
            | _ -> false

        let srcFloat = isFloatDType x.dtype
        let dstFloat = isFloatDType this.TargetDType

        if srcFloat && not dstFloat then
            let scaled = x * 255.0
            scaled.to_type this.TargetDType
        elif (not srcFloat) && dstFloat then
            let converted = x.to_type this.TargetDType
            converted / 255.0
        else
            x.to_type this.TargetDType

    interface ITransform with
        member this.apply x = this.apply x

module ConvertImageDType =
    let create (dtype: torch.ScalarType) : ConvertImageDType = { TargetDType = dtype }
