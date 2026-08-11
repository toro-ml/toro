namespace Toro.NN

open Toro

// --- Conv1d ---

/// Configuration for 1-D convolution.
type Conv1dConfig = {
    Padding: int
    Stride: int
    Dilation: int
    Groups: int
}

module Conv1dConfig =
    /// Default configuration: no padding, stride 1, dilation 1, groups 1.
    let defaultConfig = {
        Padding = 0
        Stride = 1
        Dilation = 1
        Groups = 1
    }

/// 1-D convolution layer.
type Conv1d = {
    Weight: Tensor
    Bias: Tensor option
    Config: Conv1dConfig
} with

    /// Apply 1-D convolution to the input.
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
    /// Create a Conv1d layer with bias.
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
            let! ws = Init.toParam [ outChannels; groupInC; kernelSize ] dtype device Init.defaultKaimingNormal

            let! bs = Init.toParam [ outChannels ] dtype device (Init.Uniform(-bound, bound))

            return {
                Weight = ws
                Bias = Some bs
                Config = config
            }
        }

    /// Create a Conv1d layer with default configuration.
    let initDefault
        (inChannels: int)
        (outChannels: int)
        (kernelSize: int)
        (dtype: DType)
        (device: Device)
        : Result<Conv1d, ToroError> =
        init inChannels outChannels kernelSize Conv1dConfig.defaultConfig dtype device

    /// Create a Conv1d layer without bias.
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
            let! ws = Init.toParam [ outChannels; groupInC; kernelSize ] dtype device Init.defaultKaimingNormal

            return {
                Weight = ws
                Bias = None
                Config = config
            }
        }

// --- Conv2d ---

/// Configuration for 2-D convolution.
type Conv2dConfig = {
    Padding: int
    Stride: int
    Dilation: int
    Groups: int
}

module Conv2dConfig =
    /// Default configuration: no padding, stride 1, dilation 1, groups 1.
    let defaultConfig = {
        Padding = 0
        Stride = 1
        Dilation = 1
        Groups = 1
    }

/// 2-D convolution layer.
type Conv2d = {
    Weight: Tensor
    Bias: Tensor option
    Config: Conv2dConfig
} with

    /// Apply 2-D convolution to the input.
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
    /// Create a Conv2d layer with bias.
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
            let! ws = Init.toParam [ outChannels; groupInC; kernelSize; kernelSize ] dtype device Init.defaultKaimingNormal

            let! bs = Init.toParam [ outChannels ] dtype device (Init.Uniform(-bound, bound))

            return {
                Weight = ws
                Bias = Some bs
                Config = config
            }
        }

    /// Create a Conv2d layer with default configuration.
    let initDefault
        (inChannels: int)
        (outChannels: int)
        (kernelSize: int)
        (dtype: DType)
        (device: Device)
        : Result<Conv2d, ToroError> =
        init inChannels outChannels kernelSize Conv2dConfig.defaultConfig dtype device

    /// Create a Conv2d layer without bias.
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
            let! ws = Init.toParam [ outChannels; groupInC; kernelSize; kernelSize ] dtype device Init.defaultKaimingNormal

            return {
                Weight = ws
                Bias = None
                Config = config
            }
        }

// --- ConvTranspose1d ---

/// Configuration for 1-D transposed convolution.
type ConvTranspose1dConfig = {
    Padding: int
    OutputPadding: int
    Stride: int
    Dilation: int
    Groups: int
}

module ConvTranspose1dConfig =
    let defaultConfig = {
        Padding = 0
        OutputPadding = 0
        Stride = 1
        Dilation = 1
        Groups = 1
    }

/// 1-D transposed convolution layer.
type ConvTranspose1d = {
    Weight: Tensor
    Bias: Tensor option
    Config: ConvTranspose1dConfig
} with

    member this.forward(x: Tensor) : Result<Tensor, ToroError> =
        let c = this.Config

        x.convTranspose1d (
            this.Weight,
            ?bias = this.Bias,
            stride = c.Stride,
            padding = c.Padding,
            outputPadding = c.OutputPadding,
            dilation = c.Dilation,
            groups = c.Groups
        )

    interface IModule with
        member this.forward x = this.forward x

module ConvTranspose1d =
    let init
        (inChannels: int)
        (outChannels: int)
        (kernelSize: int)
        (config: ConvTranspose1dConfig)
        (dtype: DType)
        (device: Device)
        : Result<ConvTranspose1d, ToroError> =
        let groupInC = inChannels / config.Groups
        let bound = 1.0 / sqrt (float (groupInC * kernelSize))

        result {
            let! ws =
                Init.toParam [ inChannels; outChannels / config.Groups; kernelSize ] dtype device Init.defaultKaimingNormal

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
        : Result<ConvTranspose1d, ToroError> =
        init inChannels outChannels kernelSize ConvTranspose1dConfig.defaultConfig dtype device

// --- ConvTranspose2d ---

/// Configuration for 2-D transposed convolution.
type ConvTranspose2dConfig = {
    Padding: int
    OutputPadding: int
    Stride: int
    Dilation: int
    Groups: int
}

module ConvTranspose2dConfig =
    let defaultConfig = {
        Padding = 0
        OutputPadding = 0
        Stride = 1
        Dilation = 1
        Groups = 1
    }

/// 2-D transposed convolution layer.
type ConvTranspose2d = {
    Weight: Tensor
    Bias: Tensor option
    Config: ConvTranspose2dConfig
} with

    member this.forward(x: Tensor) : Result<Tensor, ToroError> =
        let c = this.Config

        x.convTranspose2d (
            this.Weight,
            ?bias = this.Bias,
            stride = c.Stride,
            padding = c.Padding,
            outputPadding = c.OutputPadding,
            dilation = c.Dilation,
            groups = c.Groups
        )

    interface IModule with
        member this.forward x = this.forward x

module ConvTranspose2d =
    let init
        (inChannels: int)
        (outChannels: int)
        (kernelSize: int)
        (config: ConvTranspose2dConfig)
        (dtype: DType)
        (device: Device)
        : Result<ConvTranspose2d, ToroError> =
        let groupInC = inChannels / config.Groups
        let bound = 1.0 / sqrt (float (groupInC * kernelSize * kernelSize))

        result {
            let! ws =
                Init.toParam
                    [ inChannels; outChannels / config.Groups; kernelSize; kernelSize ]
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
        : Result<ConvTranspose2d, ToroError> =
        init inChannels outChannels kernelSize ConvTranspose2dConfig.defaultConfig dtype device
