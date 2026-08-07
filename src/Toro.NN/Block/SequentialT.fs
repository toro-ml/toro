namespace Toro.NN

open Toro

type SequentialT = {
    Layers: IModuleT list
} with

    member this.forwardT (x: Tensor) (train: bool) : Result<Tensor, ToroError> =
        let rec loop (layers: IModuleT list) (t: Tensor) =
            match layers with
            | [] -> Ok t
            | layer :: rest ->
                match layer.forwardT t train with
                | Ok t' -> loop rest t'
                | err -> err

        loop this.Layers x

    interface IModuleT with
        member this.forwardT x train = this.forwardT x train

module SequentialT =
    let create (layers: IModuleT list) : SequentialT = { Layers = layers }

    let ofModules (layers: IModule list) : SequentialT = {
        Layers = layers |> List.map ModuleT.ofModule
    }

type SequentialTBuilder() =
    member _.Yield(m: IModule) = [ ModuleT.ofModule m ]
    member _.Yield(m: IModuleT) = [ m ]
    member _.Combine(a: IModuleT list, b: IModuleT list) = a @ b
    member _.Delay(f: unit -> IModuleT list) = f ()
    member _.Zero() : IModuleT list = []
    member _.Run(layers: IModuleT list) = SequentialT.create layers

[<AutoOpen>]
module SequentialTCE =
    let sequentialT = SequentialTBuilder()
