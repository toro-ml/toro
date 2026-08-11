namespace Toro.NN

open TorchSharp
open Toro

type MaxPool1d = {
    KernelSize: int64
    Stride: int64
    Padding: int64
} with

    member this.forward(x: Tensor) : Tensor =
        let s = this.Stride
        let p = this.Padding
        torch.nn.functional.max_pool1d (x, this.KernelSize, stride = s, padding = p)

    interface IModule with
        member this.forward x = this.forward x

module MaxPool1d =
    let create (kernelSize: int64) (stride: int64) (padding: int64) : MaxPool1d = {
        KernelSize = kernelSize
        Stride = stride
        Padding = padding
    }

    let createDefault (kernelSize: int64) : MaxPool1d = {
        KernelSize = kernelSize
        Stride = kernelSize
        Padding = 0
    }

type MaxPool2d = {
    KernelSize: int64
    Stride: int64
    Padding: int64
} with

    member this.forward(x: Tensor) : Tensor =
        let s = this.Stride
        let p = this.Padding
        torch.nn.functional.max_pool2d (x, this.KernelSize, stride = s, padding = p)

    interface IModule with
        member this.forward x = this.forward x

module MaxPool2d =
    let create (kernelSize: int64) (stride: int64) (padding: int64) : MaxPool2d = {
        KernelSize = kernelSize
        Stride = stride
        Padding = padding
    }

    let createDefault (kernelSize: int64) : MaxPool2d = {
        KernelSize = kernelSize
        Stride = kernelSize
        Padding = 0
    }

type AvgPool2d = {
    KernelSize: int64
    Stride: int64
    Padding: int64
} with

    member this.forward(x: Tensor) : Tensor =
        let s = this.Stride
        let p = this.Padding
        torch.nn.functional.avg_pool2d (x, this.KernelSize, stride = s, padding = p)

    interface IModule with
        member this.forward x = this.forward x

module AvgPool2d =
    let create (kernelSize: int64) (stride: int64) (padding: int64) : AvgPool2d = {
        KernelSize = kernelSize
        Stride = stride
        Padding = padding
    }

    let createDefault (kernelSize: int64) : AvgPool2d = {
        KernelSize = kernelSize
        Stride = kernelSize
        Padding = 0
    }

type AvgPool1d = {
    KernelSize: int64
    Stride: int64
    Padding: int64
} with

    member this.forward(x: Tensor) : Tensor =
        let s = this.Stride
        let p = this.Padding
        torch.nn.functional.avg_pool1d (x, this.KernelSize, stride = s, padding = p)

    interface IModule with
        member this.forward x = this.forward x

module AvgPool1d =
    let create (kernelSize: int64) (stride: int64) (padding: int64) : AvgPool1d = {
        KernelSize = kernelSize
        Stride = stride
        Padding = padding
    }

    let createDefault (kernelSize: int64) : AvgPool1d = {
        KernelSize = kernelSize
        Stride = kernelSize
        Padding = 0
    }

type AdaptiveAvgPool2d = {
    OutputSize: int64
} with

    member this.forward(x: Tensor) : Tensor =
        torch.nn.functional.adaptive_avg_pool2d (x, this.OutputSize)

    interface IModule with
        member this.forward x = this.forward x

module AdaptiveAvgPool2d =
    let create (outputSize: int64) : AdaptiveAvgPool2d = { OutputSize = outputSize }

type AdaptiveAvgPool1d = {
    OutputSize: int64
} with

    member this.forward(x: Tensor) : Tensor =
        torch.nn.functional.adaptive_avg_pool1d (x, this.OutputSize)

    interface IModule with
        member this.forward x = this.forward x

module AdaptiveAvgPool1d =
    let create (outputSize: int64) : AdaptiveAvgPool1d = { OutputSize = outputSize }
