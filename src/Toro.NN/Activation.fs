namespace Toro.NN

open Toro

type Activation =
    | Relu
    | Gelu
    | Silu
    | Tanh
    | Sigmoid

module Activation =
    let apply (act: Activation) (x: Tensor) : Result<Tensor, ToroError> =
        match act with
        | Relu -> x.relu ()
        | Gelu -> x.gelu ()
        | Silu -> x.silu ()
        | Tanh -> x.tanh ()
        | Sigmoid -> x.sigmoid ()

    let toModule (act: Activation) : IModule =
        { new IModule with
            member _.forward x = apply act x
        }
