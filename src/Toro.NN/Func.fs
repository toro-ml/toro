namespace Toro.NN

open Toro

type Func =
    { F: Tensor -> Result<Tensor, ToroError> }

    interface IModule with
        member this.forward x = this.F x

module Func =
    let create (f: Tensor -> Result<Tensor, ToroError>) : Func = { F = f }

type FuncT =
    { F: Tensor -> bool -> Result<Tensor, ToroError> }

    interface IModuleT with
        member this.forwardT x train = this.F x train

module FuncT =
    let create (f: Tensor -> bool -> Result<Tensor, ToroError>) : FuncT = { F = f }

type Identity() =
    interface IModule with
        member _.forward x = Ok x
