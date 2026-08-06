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

    member this.forward(x: Tensor) : Result<Tensor, ToroError> =
        match this with
        | Relu -> x.relu ()
        | Gelu -> x.gelu ()
        | Silu -> x.silu ()
        | Tanh -> x.tanh ()
        | Sigmoid -> x.sigmoid ()
        | LeakyRelu slope -> x.leakyRelu slope
        | Elu alpha -> x.elu alpha
        | Mish -> x.mish ()

    interface IModule with
        member this.forward x = this.forward x
