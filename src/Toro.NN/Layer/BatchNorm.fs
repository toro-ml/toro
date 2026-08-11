namespace Toro.NN

open TorchSharp
open Toro

type BatchNormConfig = {
    Eps: float
    Momentum: float
    Affine: bool
}

module BatchNormConfig =
    let defaultConfig = {
        Eps = 1e-5
        Momentum = 0.1
        Affine = true
    }

type BatchNorm = {
    Weight: Tensor option
    Bias: Tensor option
    RunningMean: Tensor
    RunningVar: Tensor
    Config: BatchNormConfig
} with

    member this.forwardT (train: bool) (x: Tensor) : Tensor =
        let w = this.Weight |> Option.defaultValue null
        let b = this.Bias |> Option.defaultValue null

        torch.nn.functional.batch_norm (
            x,
            this.RunningMean,
            this.RunningVar,
            w,
            b,
            train,
            this.Config.Momentum,
            this.Config.Eps
        )

module BatchNorm =
    let init (numFeatures: int64) (config: BatchNormConfig) (dtype: torch.ScalarType) (device: torch.Device) : BatchNorm =
        let affine = if config.Affine then Some() else None

        let weight =
            affine
            |> Option.map (fun () -> Init.toParam [| numFeatures |] dtype device (Init.Const 1.0))

        let bias =
            affine
            |> Option.map (fun () -> Init.toParam [| numFeatures |] dtype device (Init.Const 0.0))

        let runningMean = torch.zeros ([| numFeatures |], dtype = dtype, device = device)

        let runningVar = torch.ones ([| numFeatures |], dtype = dtype, device = device)

        {
            Weight = weight
            Bias = bias
            RunningMean = runningMean
            RunningVar = runningVar
            Config = config
        }

    let initDefault (numFeatures: int64) (dtype: torch.ScalarType) (device: torch.Device) : BatchNorm =
        init numFeatures BatchNormConfig.defaultConfig dtype device
