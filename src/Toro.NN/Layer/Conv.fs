namespace Toro.NN

open TorchSharp
open Toro

// --- Conv1d ---

/// Configuration for 1-D convolution.
type Conv1dConfig = {
    Padding: int64
    Stride: int64
    Dilation: int64
    Groups: int64
}

module Conv1dConfig =
    /// Default configuration: no padding, stride 1, dilation 1, groups 1.
    let defaultConfig = {
        Padding = 0L
        Stride = 1L
        Dilation = 1L
        Groups = 1L
    }

/// 1-D convolution layer.
type Conv1d = {
    [<Parameter>]
    Weight: Tensor
    [<Parameter>]
    Bias: Tensor option
    Config: Conv1dConfig
} with

    /// Apply 1-D convolution to the input.
    member this.forward(x: Tensor) : Tensor =
        let c = this.Config
        let s = c.Stride
        let p = c.Padding
        let d = c.Dilation
        let g = c.Groups
        let b = this.Bias |> Option.defaultValue null
        torch.nn.functional.conv1d (x, this.Weight, b, s, p, d, g)

    interface IModule with
        member this.forward x = this.forward x

module Conv1d =
    /// Create a Conv1d layer with bias.
    let init
        (inChannels: int64)
        (outChannels: int64)
        (kernelSize: int64)
        (config: Conv1dConfig)
        (dtype: torch.ScalarType)
        (device: torch.Device)
        : Conv1d =
        let groupInC = inChannels / config.Groups
        let bound = 1.0 / sqrt (float (groupInC * kernelSize))

        let ws =
            Init.toParam [| outChannels; groupInC; kernelSize |] dtype device Init.defaultKaimingNormal

        let bs = Init.toParam [| outChannels |] dtype device (Init.Uniform(-bound, bound))

        {
            Weight = ws
            Bias = Some bs
            Config = config
        }

    /// Create a Conv1d layer with default configuration.
    let initDefault
        (inChannels: int)
        (outChannels: int)
        (kernelSize: int)
        (dtype: torch.ScalarType)
        (device: torch.Device)
        : Conv1d =
        init inChannels outChannels kernelSize Conv1dConfig.defaultConfig dtype device

    /// Create a Conv1d layer without bias.
    let initNoBias
        (inChannels: int64)
        (outChannels: int64)
        (kernelSize: int64)
        (config: Conv1dConfig)
        (dtype: torch.ScalarType)
        (device: torch.Device)
        : Conv1d =
        let groupInC = inChannels / config.Groups

        let ws =
            Init.toParam [| outChannels; groupInC; kernelSize |] dtype device Init.defaultKaimingNormal

        {
            Weight = ws
            Bias = None
            Config = config
        }

// --- Conv2d ---

/// Configuration for 2-D convolution.
type Conv2dConfig = {
    Padding: int64
    Stride: int64
    Dilation: int64
    Groups: int64
}

module Conv2dConfig =
    /// Default configuration: no padding, stride 1, dilation 1, groups 1.
    let defaultConfig = {
        Padding = 0L
        Stride = 1L
        Dilation = 1L
        Groups = 1L
    }

/// 2-D convolution layer.
type Conv2d = {
    [<Parameter>]
    Weight: Tensor
    [<Parameter>]
    Bias: Tensor option
    Config: Conv2dConfig
} with

    /// Apply 2-D convolution to the input.
    member this.forward(x: Tensor) : Tensor =
        let c = this.Config
        let s = c.Stride
        let p = c.Padding
        let d = c.Dilation
        let g = c.Groups
        let b = this.Bias |> Option.defaultValue null
        torch.nn.functional.conv2d (x, this.Weight, b, [| s; s |], [| p; p |], [| d; d |], g)

    interface IModule with
        member this.forward x = this.forward x

module Conv2d =
    /// Create a Conv2d layer with bias.
    let init
        (inChannels: int64)
        (outChannels: int64)
        (kernelSize: int64)
        (config: Conv2dConfig)
        (dtype: torch.ScalarType)
        (device: torch.Device)
        : Conv2d =
        let groupInC = inChannels / config.Groups
        let bound = 1.0 / sqrt (float (groupInC * kernelSize * kernelSize))

        let ws =
            Init.toParam [| outChannels; groupInC; kernelSize; kernelSize |] dtype device Init.defaultKaimingNormal

        let bs = Init.toParam [| outChannels |] dtype device (Init.Uniform(-bound, bound))

        {
            Weight = ws
            Bias = Some bs
            Config = config
        }

    /// Create a Conv2d layer with default configuration.
    let initDefault
        (inChannels: int64)
        (outChannels: int64)
        (kernelSize: int64)
        (dtype: torch.ScalarType)
        (device: torch.Device)
        : Conv2d =
        init inChannels outChannels kernelSize Conv2dConfig.defaultConfig dtype device

    /// Create a Conv2d layer without bias.
    let initNoBias
        (inChannels: int64)
        (outChannels: int64)
        (kernelSize: int64)
        (config: Conv2dConfig)
        (dtype: torch.ScalarType)
        (device: torch.Device)
        : Conv2d =
        let groupInC = inChannels / config.Groups

        let ws =
            Init.toParam [| outChannels; groupInC; kernelSize; kernelSize |] dtype device Init.defaultKaimingNormal

        {
            Weight = ws
            Bias = None
            Config = config
        }

// --- ConvTranspose1d ---

/// Configuration for 1-D transposed convolution.
type ConvTranspose1dConfig = {
    Padding: int64
    OutputPadding: int64
    Stride: int64
    Dilation: int64
    Groups: int64
}

module ConvTranspose1dConfig =
    let defaultConfig = {
        Padding = 0L
        OutputPadding = 0L
        Stride = 1L
        Dilation = 1L
        Groups = 1L
    }

/// 1-D transposed convolution layer.
type ConvTranspose1d = {
    [<Parameter>]
    Weight: Tensor
    [<Parameter>]
    Bias: Tensor option
    Config: ConvTranspose1dConfig
} with

    member this.forward(x: Tensor) : Tensor =
        let c = this.Config
        let s = c.Stride
        let p = c.Padding
        let op = c.OutputPadding
        let d = c.Dilation
        let g = c.Groups
        let b = this.Bias |> Option.defaultValue null
        torch.nn.functional.conv_transpose1d (x, this.Weight, b, s, p, op, g, d)

    interface IModule with
        member this.forward x = this.forward x

module ConvTranspose1d =
    let init
        (inChannels: int64)
        (outChannels: int64)
        (kernelSize: int64)
        (config: ConvTranspose1dConfig)
        (dtype: torch.ScalarType)
        (device: torch.Device)
        : ConvTranspose1d =
        let groupInC = inChannels / config.Groups
        let bound = 1.0 / sqrt (float (groupInC * kernelSize))

        let ws =
            Init.toParam [| inChannels; (outChannels / config.Groups); kernelSize |] dtype device Init.defaultKaimingNormal

        let bs = Init.toParam [| outChannels |] dtype device (Init.Uniform(-bound, bound))

        {
            Weight = ws
            Bias = Some bs
            Config = config
        }

    let initDefault
        (inChannels: int64)
        (outChannels: int64)
        (kernelSize: int64)
        (dtype: torch.ScalarType)
        (device: torch.Device)
        : ConvTranspose1d =
        init inChannels outChannels kernelSize ConvTranspose1dConfig.defaultConfig dtype device

// --- ConvTranspose2d ---

/// Configuration for 2-D transposed convolution.
type ConvTranspose2dConfig = {
    Padding: int64
    OutputPadding: int64
    Stride: int64
    Dilation: int64
    Groups: int64
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
    [<Parameter>]
    Weight: Tensor
    [<Parameter>]
    Bias: Tensor option
    Config: ConvTranspose2dConfig
} with

    member this.forward(x: Tensor) : Tensor =
        let c = this.Config
        let s = int64 c.Stride
        let p = int64 c.Padding
        let op = int64 c.OutputPadding
        let d = int64 c.Dilation
        let g = int64 c.Groups
        let b = this.Bias |> Option.defaultValue null

        torch.nn.functional.conv_transpose2d (x, this.Weight, b, [| s; s |], [| p; p |], [| op; op |], [| d; d |], g)

    interface IModule with
        member this.forward x = this.forward x

module ConvTranspose2d =
    let init
        (inChannels: int64)
        (outChannels: int64)
        (kernelSize: int64)
        (config: ConvTranspose2dConfig)
        (dtype: torch.ScalarType)
        (device: torch.Device)
        : ConvTranspose2d =
        let groupInC = inChannels / config.Groups
        let bound = 1.0 / sqrt (float (groupInC * kernelSize * kernelSize))

        let ws =
            Init.toParam
                [| inChannels; outChannels / config.Groups; kernelSize; kernelSize |]
                dtype
                device
                Init.defaultKaimingNormal

        let bs = Init.toParam [| outChannels |] dtype device (Init.Uniform(-bound, bound))

        {
            Weight = ws
            Bias = Some bs
            Config = config
        }

    let initDefault
        (inChannels: int64)
        (outChannels: int64)
        (kernelSize: int64)
        (dtype: torch.ScalarType)
        (device: torch.Device)
        : ConvTranspose2d =
        init inChannels outChannels kernelSize ConvTranspose2dConfig.defaultConfig dtype device
