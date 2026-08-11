namespace Toro.NN

open Toro

type Func = {
    F: Tensor -> Tensor
} with

    interface IModule with
        member this.forward x = this.F x

module Func =
    let create (f: Tensor -> Tensor) : Func = { F = f }

    let Identity: IModule = create id :> IModule

type PipelineBuilder() =
    member _.Yield(m: #IModule<'a, 'b>) : 'a -> 'b = m.forward
    member _.Yield(f: 'a -> 'b) : 'a -> 'b = f
    member inline _.Combine(f, g) = f >> g
    member _.Delay(f) = f ()
    member _.Zero() = id

[<AutoOpen>]
module PipelineCE =
    let pipeline = PipelineBuilder()
