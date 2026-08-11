namespace Toro.NN

open TorchSharp
open Toro

type Activation =
    | Relu
    | Gelu
    | Silu
    | Tanh
    | Sigmoid
    | LeakyRelu of negativeSlope: float
    | Elu of alpha: float
    | Mish
    | Celu of alpha: float
    | Selu
    | Glu of dim: int64
    | Hardswish
    | Hardsigmoid

    member this.forward(x: Tensor) : Tensor =
        match this with
        | Relu -> x.relu ()
        | Gelu -> x.gelu ()
        | Silu -> x.silu ()
        | Tanh -> x.tanh ()
        | Sigmoid -> x.sigmoid ()
        | LeakyRelu slope -> torch.nn.functional.leaky_relu (x, slope)
        | Elu alpha -> x.elu alpha
        | Mish -> torch.nn.functional.mish x
        | Celu alpha -> x.celu alpha
        | Selu -> x.selu ()
        | Glu dim -> x.glu dim
        | Hardswish -> x.hardswish ()
        | Hardsigmoid -> x.hardsigmoid ()

    interface IModule with
        member this.forward x = this.forward x
