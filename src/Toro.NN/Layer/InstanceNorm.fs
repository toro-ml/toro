namespace Toro.NN

open TorchSharp
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
    [<Parameter>]
    Weight: Tensor option
    [<Parameter>]
    Bias: Tensor option
    Eps: float
    Momentum: float
} with

    member this.forward(x: Tensor) : Tensor =
        let w = this.Weight |> Option.defaultValue null
        let b = this.Bias |> Option.defaultValue null
        torch.nn.functional.instance_norm (x, null, null, w, b, true, this.Momentum, this.Eps)

    interface IModule with
        member this.forward x = this.forward x

module InstanceNorm =
    let init (numFeatures: int64) (config: InstanceNormConfig) (dtype: torch.ScalarType) (device: torch.Device) : InstanceNorm =
        let weight =
            (if config.Affine then Some(Init.Const 1.0) else None)
            |> Option.map (Init.toParam [| numFeatures |] dtype device)

        let bias =
            (if config.Affine then Some(Init.Const 0.0) else None)
            |> Option.map (Init.toParam [| numFeatures |] dtype device)

        {
            Weight = weight
            Bias = bias
            Eps = config.Eps
            Momentum = config.Momentum
        }

    let initDefault (numFeatures: int64) (dtype: torch.ScalarType) (device: torch.Device) : InstanceNorm =
        init numFeatures InstanceNormConfig.defaultConfig dtype device
