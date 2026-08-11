namespace Toro.NN

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
    | Glu of dim: int
    | Hardswish
    | Hardsigmoid

    member this.forward(x: Tensor) : Tensor =
        match this with
        | Relu -> x.relu ()
        | Gelu -> x.gelu ()
        | Silu -> x.silu ()
        | Tanh -> x.tanh ()
        | Sigmoid -> x.sigmoid ()
        | LeakyRelu slope -> x.leakyRelu slope
        | Elu alpha -> x.elu alpha
        | Mish -> x.mish ()
        | Celu alpha -> x.celu alpha
        | Selu -> x.selu ()
        | Glu dim -> x.glu dim
        | Hardswish -> x.hardswish ()
        | Hardsigmoid -> x.hardsigmoid ()

    interface IModule with
        member this.forward x = this.forward x
