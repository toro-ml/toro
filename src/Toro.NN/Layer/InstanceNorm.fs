namespace Toro.NN

open Toro

type InstanceNormConfig = {
    Eps: float
    Momentum: float
    Affine: bool
}

module InstanceNormConfig =
    let defaultConfig = {
        Eps = 1e-5
        Momentum = 0.1
        Affine = false
    }

/// Instance normalization layer.
type InstanceNorm = {
    Weight: Tensor option
    Bias: Tensor option
    Eps: float
    Momentum: float
} with

    member this.forward(x: Tensor) : Result<Tensor, ToroError> =
        x.instanceNorm (?weight = this.Weight, ?bias = this.Bias, momentum = this.Momentum, eps = this.Eps)

    interface IModule with
        member this.forward x = this.forward x

module InstanceNorm =
    let init (numFeatures: int) (config: InstanceNormConfig) (dtype: DType) (device: Device) : Result<InstanceNorm, ToroError> =
        result {
            let! weight =
                (if config.Affine then Some(Init.Const 1.0) else None)
                |> Option.traverseResult (Init.toParam [ numFeatures ] dtype device)

            let! bias =
                (if config.Affine then Some(Init.Const 0.0) else None)
                |> Option.traverseResult (Init.toParam [ numFeatures ] dtype device)

            return {
                Weight = weight
                Bias = bias
                Eps = config.Eps
                Momentum = config.Momentum
            }
        }

    let initDefault (numFeatures: int) (dtype: DType) (device: Device) : Result<InstanceNorm, ToroError> =
        init numFeatures InstanceNormConfig.defaultConfig dtype device
