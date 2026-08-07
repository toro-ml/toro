namespace Toro.NN

open Toro

type Sequential = {
    Layers: IModule list
} with

    member this.forward(x: Tensor) : Result<Tensor, ToroError> =
        let rec loop (layers: IModule list) (t: Tensor) =
            match layers with
            | [] -> Ok t
            | layer :: rest ->
                match layer.forward t with
                | Ok t' -> loop rest t'
                | err -> err

        loop this.Layers x

    interface IModule with
        member this.forward x = this.forward x

module Sequential =
    let create (layers: IModule list) : Sequential = { Layers = layers }

type SequentialBuilder() =
    member _.Yield(m: #IModule) = [ m :> IModule ]
    member _.Combine(a: IModule list, b: IModule list) = a @ b
    member _.Delay(f: unit -> IModule list) = f ()
    member _.Zero() : IModule list = []
    member _.Run(layers: IModule list) = Sequential.create layers

[<AutoOpen>]
module SequentialCE =
    let sequential = SequentialBuilder()
