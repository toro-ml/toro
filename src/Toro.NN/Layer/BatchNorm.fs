namespace Toro.NN

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

    member this.forwardT (train: bool) (x: Tensor) : Result<Tensor, ToroError> =
        x.batchNorm (
            this.Weight,
            this.Bias,
            Some this.RunningMean,
            Some this.RunningVar,
            train,
            this.Config.Momentum,
            this.Config.Eps
        )

module BatchNorm =
    let init (numFeatures: int) (config: BatchNormConfig) (dtype: DType) (device: Device) : Result<BatchNorm, ToroError> =
        result {
            let affine = if config.Affine then Some() else None

            let! weight =
                affine
                |> Option.traverseResult (fun () -> Init.toParam [ numFeatures ] dtype device (Init.Const 1.0))

            let! bias =
                affine
                |> Option.traverseResult (fun () -> Init.toParam [ numFeatures ] dtype device (Init.Const 0.0))

            let! runningMean = Tensor.zeros ([ numFeatures ], dtype, device)
            let! runningVar = Tensor.ones ([ numFeatures ], dtype, device)

            return {
                Weight = weight
                Bias = bias
                RunningMean = runningMean
                RunningVar = runningVar
                Config = config
            }
        }

    let initDefault (numFeatures: int) (dtype: DType) (device: Device) : Result<BatchNorm, ToroError> =
        init numFeatures BatchNormConfig.defaultConfig dtype device
