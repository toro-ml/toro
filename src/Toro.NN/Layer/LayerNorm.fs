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

    member this.forward(x: Tensor) : Result<Tensor, ToroError> =
        result {
            let xDType = x.DType

            let internalDType =
                match xDType with
                | F16
                | BF16 -> F32
                | d -> d

            let! x = x.toDType internalDType

            let! x =
                if this.RemoveMean then
                    result {
                        let! meanX = x.meanKeepdim -1
                        return! x.sub meanX
                    }
                else
                    Ok x

            let! xSqr = x.sqr ()
            let! normX = xSqr.meanKeepdim -1
            let! xNormed = x /~ (normX.addScalar this.Eps |> TensorR.sqrt)
            let! xNormed = xNormed.toDType xDType

            let! x = xNormed.mul this.Weight

            match this.Bias with
            | None -> return x
            | Some bias -> return! x.add bias
        }

    interface IModule with
        member this.forward x = this.forward x

module LayerNorm =
    let init (size: int) (config: LayerNormConfig) (dtype: DType) (device: Device) : Result<LayerNorm, ToroError> =
        result {
            let! weight = Init.toParam [ size ] dtype device (Init.Const 1.0)

            let! bias =
                (if config.Affine then Some(Init.Const 0.0) else None)
                |> Option.traverseResult (Init.toParam [ size ] dtype device)

            return {
                Weight = weight
                Bias = bias
                RemoveMean = config.RemoveMean
                Eps = config.Eps
            }
        }

    let initDefault (size: int) (dtype: DType) (device: Device) : Result<LayerNorm, ToroError> =
        init size LayerNormConfig.defaultConfig dtype device

type RmsNorm = {
    Inner: LayerNorm
} with

    member this.forward(x: Tensor) : Result<Tensor, ToroError> = this.Inner.forward x

    interface IModule with
        member this.forward x = this.forward x

module RmsNorm =
    let init (size: int) (eps: float) (dtype: DType) (device: Device) : Result<RmsNorm, ToroError> =
        let config = {
            Eps = eps
            RemoveMean = false
            Affine = false
        }

        result {
            let! inner = LayerNorm.init size config dtype device
            return { Inner = inner }
        }
