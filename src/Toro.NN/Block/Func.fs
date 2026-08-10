namespace Toro.NN

open Toro

type Func = {
    F: Tensor -> Result<Tensor, ToroError>
} with

    interface IModule with
        member this.forward x = this.F x

module Func =
    let create (f: Tensor -> Result<Tensor, ToroError>) : Func = { F = f }

    let Identity: IModule = create Ok :> IModule

type PipelineBuilder() =
    member _.Yield(m: #IModule<'a, 'b>) : 'a -> Result<'b, ToroError> = m.forward
    member _.Yield(f: 'a -> Result<'b, ToroError>) : 'a -> Result<'b, ToroError> = f
    member inline _.Combine(f, g) = f >=> g
    member _.Delay(f) = f ()
    member _.Zero() = Ok

[<AutoOpen>]
module PipelineCE =
    let pipeline = PipelineBuilder()
