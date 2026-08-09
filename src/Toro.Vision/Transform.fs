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

    interface ITransform with
        member this.apply x =
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

module Normalize =
    /// ImageNet normalization preset.
    let imageNet: ITransform = {
        Mean = [ 0.485; 0.456; 0.406 ]
        Std = [ 0.229; 0.224; 0.225 ]
    }

/// Resize to the target spatial size using bilinear interpolation.
/// Input: $[C, H, W]$ or $[B, C, H, W]$.
type Resize = {
    Height: int
    Width: int
} with

    interface ITransform with
        member this.apply x =
            let rank = x.Rank

            result {
                if rank = 3 then
                    let! batched = x.unsqueeze 0
                    let! resized = batched.interpolate ([ this.Height; this.Width ], Bilinear)
                    return! resized.squeeze 0
                else
                    return! x.interpolate ([ this.Height; this.Width ], Bilinear)
            }

module Resize =
    let create (height: int) (width: int) : ITransform = { Height = height; Width = width }

/// Randomly flip the image horizontally with probability p.
/// Input: $[C, H, W]$ or $[B, C, H, W]$.
type RandomHorizontalFlip = {
    P: float
} with

    interface ITransform with
        member this.apply x =
            let wDim = x.Rank - 1

            if System.Random.Shared.NextDouble() < this.P then
                x.flip [ wDim ]
            else
                Ok x

module RandomHorizontalFlip =
    let create (p: float) : ITransform = { P = p }
    let defaultFlip: ITransform = { P = 0.5 }

/// Randomly crop the image to the given size.
/// Input: $[C, H, W]$ or $[B, C, H, W]$.
type RandomCrop = {
    Height: int
    Width: int
} with

    interface ITransform with
        member this.apply x =
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

module RandomCrop =
    let create (height: int) (width: int) : ITransform = { Height = height; Width = width }
