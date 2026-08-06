namespace Toro.NN

open Toro

type Activation =
    | Relu
    | Gelu
    | Silu
    | Tanh
    | Sigmoid

    member this.forward(x: Tensor) : Result<Tensor, ToroError> =
        match this with
        | Relu -> x.relu ()
        | Gelu -> x.gelu ()
        | Silu -> x.silu ()
        | Tanh -> x.tanh ()
        | Sigmoid -> x.sigmoid ()

    interface IModule with
        member this.forward x = this.forward x
