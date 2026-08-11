namespace Toro.NN

open Toro

type GroupNormConfig = { Eps: float; Affine: bool }

module GroupNormConfig =
    let defaultConfig = { Eps = 1e-5; Affine = true }

type GroupNorm = {
    NumGroups: int
    Weight: Tensor option
    Bias: Tensor option
    Eps: float
} with

    member this.forward(x: Tensor) : Tensor =
        x.groupNorm (this.NumGroups, ?weight = this.Weight, ?bias = this.Bias, eps = this.Eps)

    interface IModule with
        member this.forward x = this.forward x

module GroupNorm =
    let init (numGroups: int) (numChannels: int) (config: GroupNormConfig) (dtype: DType) (device: Device) : GroupNorm =
        let affine = if config.Affine then Some() else None

        let weight =
            affine
            |> Option.map (fun () -> Init.toParam [ numChannels ] dtype device (Init.Const 1.0))

        let bias =
            affine
            |> Option.map (fun () -> Init.toParam [ numChannels ] dtype device (Init.Const 0.0))

        {
            NumGroups = numGroups
            Weight = weight
            Bias = bias
            Eps = config.Eps
        }

    let initDefault (numGroups: int) (numChannels: int) (dtype: DType) (device: Device) : GroupNorm =
        init numGroups numChannels GroupNormConfig.defaultConfig dtype device
