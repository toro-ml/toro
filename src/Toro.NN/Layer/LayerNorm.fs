namespace Toro.NN

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
            x.layerNorm (this.Weight.Shape, weight = this.Weight, ?bias = this.Bias, eps = this.Eps)
        else
            let xDType = x.DType

            let internalDType =
                match xDType with
                | F16
                | BF16 -> F32
                | d -> d

            let x = x.toDType internalDType
            let xSqr = x.sqr ()
            let normX = xSqr.mean (-1, keepDim = true)
            let xNormed = x / (normX.addScalar(this.Eps).sqrt ())
            let xNormed = xNormed.toDType xDType
            let x = xNormed.mul this.Weight

            match this.Bias with
            | None -> x
            | Some bias -> x.add bias

    interface IModule with
        member this.forward x = this.forward x

module LayerNorm =
    let init (size: int) (config: LayerNormConfig) (dtype: DType) (device: Device) : LayerNorm =
        let weight = Init.toParam [ size ] dtype device (Init.Const 1.0)

        let bias =
            (if config.Affine then Some(Init.Const 0.0) else None)
            |> Option.map (Init.toParam [ size ] dtype device)

        {
            Weight = weight
            Bias = bias
            RemoveMean = config.RemoveMean
            Eps = config.Eps
        }

    let initDefault (size: int) (dtype: DType) (device: Device) : LayerNorm =
        init size LayerNormConfig.defaultConfig dtype device

type RmsNorm = {
    Inner: LayerNorm
} with

    member this.forward(x: Tensor) : Tensor = this.Inner.forward x

    interface IModule with
        member this.forward x = this.forward x

module RmsNorm =
    let init (size: int) (eps: float) (dtype: DType) (device: Device) : RmsNorm =
        let config = {
            Eps = eps
            RemoveMean = false
            Affine = false
        }

        let inner = LayerNorm.init size config dtype device
        { Inner = inner }
