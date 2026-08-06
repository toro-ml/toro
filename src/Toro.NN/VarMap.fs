namespace Toro.NN

open System.Collections.Generic
open System.IO
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

    member _.save(dirPath: string) : Result<unit, ToroError> =
        result {
            do! ToroError.wrap (fun () -> Directory.CreateDirectory(dirPath) |> ignore)

            for kv in data do
                let filePath = Path.Combine(dirPath, kv.Key + ".toro")
                let dir = Path.GetDirectoryName(filePath)

                if not (Directory.Exists(dir)) then
                    Directory.CreateDirectory(dir) |> ignore

                do! kv.Value.save filePath
        }

    static member load(dirPath: string) : Result<VarMap, ToroError> =
        result {
            let! files =
                ToroError.wrap (fun () -> Directory.GetFiles(dirPath, "*.toro", SearchOption.AllDirectories))

            let vm = VarMap()

            for file in files do
                let rel = Path.GetRelativePath(dirPath, file)
                let name = rel.Replace(Path.DirectorySeparatorChar, '.').Replace(".toro", "")
                let! t = Tensor.load file
                let! t = t.requiresGrad ()
                vm.set name t

            return vm
        }

    member internal _.set (name: string) (t: Tensor) = data.[name] <- t

    interface IVarBackend with
        member this.get(shape, name, init, dtype, device) =
            this.get shape name init dtype device

        member _.containsTensor name = data.ContainsKey name
