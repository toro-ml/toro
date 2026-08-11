namespace Toro.NN

open TorchSharp
open Toro

type GroupNormConfig = { Eps: float; Affine: bool }

module GroupNormConfig =
    let defaultConfig = { Eps = 1e-5; Affine = true }

type GroupNorm = {
    NumGroups: int64
    Weight: Tensor option
    Bias: Tensor option
    Eps: float
} with

    member this.forward(x: Tensor) : Tensor =
        let w = this.Weight |> Option.defaultValue null
        let b = this.Bias |> Option.defaultValue null
        torch.nn.functional.group_norm (x, this.NumGroups, w, b, this.Eps)

    interface IModule with
        member this.forward x = this.forward x

module GroupNorm =
    let init
        (numGroups: int64)
        (numChannels: int64)
        (config: GroupNormConfig)
        (dtype: torch.ScalarType)
        (device: torch.Device)
        : GroupNorm =
        let affine = if config.Affine then Some() else None

        let weight =
            affine
            |> Option.map (fun () -> Init.toParam [| numChannels |] dtype device (Init.Const 1.0))

        let bias =
            affine
            |> Option.map (fun () -> Init.toParam [| numChannels |] dtype device (Init.Const 0.0))

        {
            NumGroups = numGroups
            Weight = weight
            Bias = bias
            Eps = config.Eps
        }

    let initDefault (numGroups: int) (numChannels: int) (dtype: torch.ScalarType) (device: torch.Device) : GroupNorm =
        init numGroups numChannels GroupNormConfig.defaultConfig dtype device
