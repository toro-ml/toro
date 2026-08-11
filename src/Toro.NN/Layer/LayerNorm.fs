namespace Toro.NN

open TorchSharp
open Toro

type LayerNormConfig = {
    Eps: float
    RemoveMean: bool
    Affine: bool
}

module LayerNormConfig =
    let defaultConfig = {
        Eps = 1e-5
        RemoveMean = true
        Affine = true
    }

type LayerNorm = {
    Weight: Tensor
    Bias: Tensor option
    RemoveMean: bool
    Eps: float
} with

    member this.forward(x: Tensor) : Tensor =
        if this.RemoveMean then
            let w = this.Weight
            let b = this.Bias |> Option.defaultValue null
            torch.nn.functional.layer_norm (x, this.Weight.shape, w, b, this.Eps)
        else
            let xDType = x.dtype

            let internalDType =
                match xDType with
                | torch.ScalarType.Float16
                | torch.ScalarType.BFloat16 -> torch.ScalarType.Float32
                | d -> d

            let x = x.to_type internalDType
            let xSqr = x.square ()
            let normX = torch.mean (xSqr, [| -1L |], keepdim = true)
            let xNormed = x / (normX + scalar this.Eps).sqrt ()
            let xNormed = xNormed.to_type xDType
            let x = xNormed.mul this.Weight

            match this.Bias with
            | None -> x
            | Some bias -> x.add bias

    interface IModule with
        member this.forward x = this.forward x

module LayerNorm =
    let init (size: int64) (config: LayerNormConfig) (dtype: torch.ScalarType) (device: torch.Device) : LayerNorm =
        let weight = Init.toParam [| size |] dtype device (Init.Const 1.0)

        let bias =
            (if config.Affine then Some(Init.Const 0.0) else None)
            |> Option.map (Init.toParam [| size |] dtype device)

        {
            Weight = weight
            Bias = bias
            RemoveMean = config.RemoveMean
            Eps = config.Eps
        }

    let initDefault (size: int64) (dtype: torch.ScalarType) (device: torch.Device) : LayerNorm =
        init size LayerNormConfig.defaultConfig dtype device

type RmsNorm = {
    Inner: LayerNorm
} with

    member this.forward(x: Tensor) : Tensor = this.Inner.forward x

    interface IModule with
        member this.forward x = this.forward x

module RmsNorm =
    let init (size: int64) (eps: float) (dtype: torch.ScalarType) (device: torch.Device) : RmsNorm =
        let config = {
            Eps = eps
            RemoveMean = false
            Affine = false
        }

        let inner = LayerNorm.init size config dtype device
        { Inner = inner }
