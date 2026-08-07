namespace Toro.NN

open Toro

type MaxPool1d = {
    KernelSize: int
    Stride: int
    Padding: int
} with

    member this.forward(x: Tensor) : Result<Tensor, ToroError> =
        x.maxPool1d (this.KernelSize, stride = this.Stride, padding = this.Padding)

    interface IModule with
        member this.forward x = this.forward x

module MaxPool1d =
    let create (kernelSize: int) (stride: int) (padding: int) : MaxPool1d = {
        KernelSize = kernelSize
        Stride = stride
        Padding = padding
    }

    let createDefault (kernelSize: int) : MaxPool1d = {
        KernelSize = kernelSize
        Stride = kernelSize
        Padding = 0
    }

type MaxPool2d = {
    KernelSize: int
    Stride: int
    Padding: int
} with

    member this.forward(x: Tensor) : Result<Tensor, ToroError> =
        x.maxPool2d (this.KernelSize, stride = this.Stride, padding = this.Padding)

    interface IModule with
        member this.forward x = this.forward x

module MaxPool2d =
    let create (kernelSize: int) (stride: int) (padding: int) : MaxPool2d = {
        KernelSize = kernelSize
        Stride = stride
        Padding = padding
    }

    let createDefault (kernelSize: int) : MaxPool2d = {
        KernelSize = kernelSize
        Stride = kernelSize
        Padding = 0
    }

type AvgPool2d = {
    KernelSize: int
    Stride: int
    Padding: int
} with

    member this.forward(x: Tensor) : Result<Tensor, ToroError> =
        x.avgPool2d (this.KernelSize, stride = this.Stride, padding = this.Padding)

    interface IModule with
        member this.forward x = this.forward x

module AvgPool2d =
    let create (kernelSize: int) (stride: int) (padding: int) : AvgPool2d = {
        KernelSize = kernelSize
        Stride = stride
        Padding = padding
    }

    let createDefault (kernelSize: int) : AvgPool2d = {
        KernelSize = kernelSize
        Stride = kernelSize
        Padding = 0
    }
