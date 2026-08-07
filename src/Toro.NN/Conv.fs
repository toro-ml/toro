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
    let init
        (inChannels: int)
        (outChannels: int)
        (kernelSize: int)
        (config: Conv1dConfig)
        (dtype: DType)
        (device: Device)
        : Result<Conv1d, ToroError> =
        let groupInC = inChannels / config.Groups
        let bound = 1.0 / sqrt (float (groupInC * kernelSize))

        result {
            let! ws =
                Init.toParam [ outChannels; groupInC; kernelSize ] dtype device Init.defaultKaimingNormal

            let! bs = Init.toParam [ outChannels ] dtype device (Init.Uniform(-bound, bound))

            return {
                Weight = ws
                Bias = Some bs
                Config = config
            }
        }

    let initDefault
        (inChannels: int)
        (outChannels: int)
        (kernelSize: int)
        (dtype: DType)
        (device: Device)
        : Result<Conv1d, ToroError> =
        init inChannels outChannels kernelSize Conv1dConfig.defaultConfig dtype device

    let initNoBias
        (inChannels: int)
        (outChannels: int)
        (kernelSize: int)
        (config: Conv1dConfig)
        (dtype: DType)
        (device: Device)
        : Result<Conv1d, ToroError> =
        let groupInC = inChannels / config.Groups

        result {
            let! ws =
                Init.toParam [ outChannels; groupInC; kernelSize ] dtype device Init.defaultKaimingNormal

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
    let init
        (inChannels: int)
        (outChannels: int)
        (kernelSize: int)
        (config: Conv2dConfig)
        (dtype: DType)
        (device: Device)
        : Result<Conv2d, ToroError> =
        let groupInC = inChannels / config.Groups
        let bound = 1.0 / sqrt (float (groupInC * kernelSize * kernelSize))

        result {
            let! ws =
                Init.toParam
                    [ outChannels; groupInC; kernelSize; kernelSize ]
                    dtype
                    device
                    Init.defaultKaimingNormal

            let! bs = Init.toParam [ outChannels ] dtype device (Init.Uniform(-bound, bound))

            return {
                Weight = ws
                Bias = Some bs
                Config = config
            }
        }

    let initDefault
        (inChannels: int)
        (outChannels: int)
        (kernelSize: int)
        (dtype: DType)
        (device: Device)
        : Result<Conv2d, ToroError> =
        init inChannels outChannels kernelSize Conv2dConfig.defaultConfig dtype device

    let initNoBias
        (inChannels: int)
        (outChannels: int)
        (kernelSize: int)
        (config: Conv2dConfig)
        (dtype: DType)
        (device: Device)
        : Result<Conv2d, ToroError> =
        let groupInC = inChannels / config.Groups

        result {
            let! ws =
                Init.toParam
                    [ outChannels; groupInC; kernelSize; kernelSize ]
                    dtype
                    device
                    Init.defaultKaimingNormal

            return {
                Weight = ws
                Bias = None
                Config = config
            }
        }
