namespace Toro.NN

open Toro

// --- Conv1d ---

type Conv1dConfig = {
    Padding: int
    Stride: int
    Dilation: int
    Groups: int
}

module Conv1dConfig =
    let defaultConfig = {
        Padding = 0
        Stride = 1
        Dilation = 1
        Groups = 1
    }

type Conv1d = {
    Weight: Tensor
    Bias: Tensor option
    Config: Conv1dConfig
} with

    member this.forward(x: Tensor) : Result<Tensor, ToroError> =
        let c = this.Config

        x.conv1d (
            this.Weight,
            ?bias = this.Bias,
            stride = c.Stride,
            padding = c.Padding,
            dilation = c.Dilation,
            groups = c.Groups
        )

    interface IModule with
        member this.forward x = this.forward x

module Conv1d =
    let create
        (inChannels: int)
        (outChannels: int)
        (kernelSize: int)
        (config: Conv1dConfig)
        (vb: VarBuilder)
        : Result<Conv1d, ToroError> =
        let groupInC = inChannels / config.Groups
        let initWs = Init.defaultKaimingNormal
        let bound = 1.0 / sqrt (float (groupInC * kernelSize))
        let initBs = Init.Uniform(-bound, bound)

        result {
            let! ws =
                VarBuilder.getWithHints [ outChannels; groupInC; kernelSize ] "weight" initWs vb

            let! bs = VarBuilder.getWithHints [ outChannels ] "bias" initBs vb

            return {
                Weight = ws
                Bias = Some bs
                Config = config
            }
        }

    let createDefault
        (inChannels: int)
        (outChannels: int)
        (kernelSize: int)
        (vb: VarBuilder)
        : Result<Conv1d, ToroError> =
        create inChannels outChannels kernelSize Conv1dConfig.defaultConfig vb

    let createNoBias
        (inChannels: int)
        (outChannels: int)
        (kernelSize: int)
        (config: Conv1dConfig)
        (vb: VarBuilder)
        : Result<Conv1d, ToroError> =
        let groupInC = inChannels / config.Groups
        let initWs = Init.defaultKaimingNormal

        result {
            let! ws =
                VarBuilder.getWithHints [ outChannels; groupInC; kernelSize ] "weight" initWs vb

            return {
                Weight = ws
                Bias = None
                Config = config
            }
        }

// --- Conv2d ---

type Conv2dConfig = {
    Padding: int
    Stride: int
    Dilation: int
    Groups: int
}

module Conv2dConfig =
    let defaultConfig = {
        Padding = 0
        Stride = 1
        Dilation = 1
        Groups = 1
    }

type Conv2d = {
    Weight: Tensor
    Bias: Tensor option
    Config: Conv2dConfig
} with

    member this.forward(x: Tensor) : Result<Tensor, ToroError> =
        let c = this.Config

        x.conv2d (
            this.Weight,
            ?bias = this.Bias,
            stride = c.Stride,
            padding = c.Padding,
            dilation = c.Dilation,
            groups = c.Groups
        )

    interface IModule with
        member this.forward x = this.forward x

module Conv2d =
    let create
        (inChannels: int)
        (outChannels: int)
        (kernelSize: int)
        (config: Conv2dConfig)
        (vb: VarBuilder)
        : Result<Conv2d, ToroError> =
        let groupInC = inChannels / config.Groups
        let initWs = Init.defaultKaimingNormal
        let bound = 1.0 / sqrt (float (groupInC * kernelSize * kernelSize))
        let initBs = Init.Uniform(-bound, bound)

        result {
            let! ws =
                VarBuilder.getWithHints
                    [ outChannels; groupInC; kernelSize; kernelSize ]
                    "weight"
                    initWs
                    vb

            let! bs = VarBuilder.getWithHints [ outChannels ] "bias" initBs vb

            return {
                Weight = ws
                Bias = Some bs
                Config = config
            }
        }

    let createDefault
        (inChannels: int)
        (outChannels: int)
        (kernelSize: int)
        (vb: VarBuilder)
        : Result<Conv2d, ToroError> =
        create inChannels outChannels kernelSize Conv2dConfig.defaultConfig vb

    let createNoBias
        (inChannels: int)
        (outChannels: int)
        (kernelSize: int)
        (config: Conv2dConfig)
        (vb: VarBuilder)
        : Result<Conv2d, ToroError> =
        let groupInC = inChannels / config.Groups
        let initWs = Init.defaultKaimingNormal

        result {
            let! ws =
                VarBuilder.getWithHints
                    [ outChannels; groupInC; kernelSize; kernelSize ]
                    "weight"
                    initWs
                    vb

            return {
                Weight = ws
                Bias = None
                Config = config
            }
        }
