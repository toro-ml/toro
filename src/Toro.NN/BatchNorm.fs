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

    member this.forwardT (x: Tensor) (train: bool) : Result<Tensor, ToroError> =
        x.batchNorm (
            this.Weight,
            this.Bias,
            Some this.RunningMean,
            Some this.RunningVar,
            train,
            this.Config.Momentum,
            this.Config.Eps
        )

    interface IModuleT with
        member this.forwardT x train = this.forwardT x train

module BatchNorm =
    let create
        (numFeatures: int)
        (config: BatchNormConfig)
        (vb: VarBuilder)
        : Result<BatchNorm, ToroError> =
        result {
            let! weight, bias =
                if config.Affine then
                    result {
                        let! w =
                            VarBuilder.getWithHints
                                [ numFeatures ]
                                "weight"
                                (Init.Const 1.0)
                                vb

                        let! b =
                            VarBuilder.getWithHints
                                [ numFeatures ]
                                "bias"
                                (Init.Const 0.0)
                                vb

                        return Some w, Some b
                    }
                else
                    Ok(None, None)

            let! runningMean = Tensor.zeros ([ numFeatures ], vb.DType, vb.Device)
            let! runningVar = Tensor.ones ([ numFeatures ], vb.DType, vb.Device)

            return {
                Weight = weight
                Bias = bias
                RunningMean = runningMean
                RunningVar = runningVar
                Config = config
            }
        }

    let createDefault
        (numFeatures: int)
        (vb: VarBuilder)
        : Result<BatchNorm, ToroError> =
        create numFeatures BatchNormConfig.defaultConfig vb
