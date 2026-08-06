namespace Toro.NN

open System.Collections.Generic
open Toro

type VarMap(data: Dictionary<string, Tensor>) =

    new() = VarMap(Dictionary())

    member _.allVars() : Tensor list =
        data.Values |> Seq.toList

    member _.data() : IReadOnlyDictionary<string, Tensor> =
        data :> IReadOnlyDictionary<_, _>

    member _.get
        (shape: int list)
        (name: string)
        (init: Init)
        (dtype: DType)
        (device: Device)
        : Result<Tensor, ToroError> =
        match data.TryGetValue(name) with
        | true, t ->
            let tShape = t.Shape

            if tShape <> shape then
                Error(
                    ShapeMismatch(
                        $"shape mismatch for {name}",
                        shape,
                        tShape
                    )
                )
            else
                Ok t
        | false, _ ->
            result {
                let! t = Init.toTensor shape dtype device init
                let! t = t.requiresGrad ()
                data.[name] <- t
                return t
            }

    interface IVarBackend with
        member this.get(shape, name, init, dtype, device) =
            this.get shape name init dtype device

        member _.containsTensor name = data.ContainsKey name
