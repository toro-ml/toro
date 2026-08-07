namespace Toro.NN

open System.IO
open System.Reflection
open Microsoft.FSharp.Reflection
open Toro

module Model =

    let private tensorType = typeof<Tensor>

    let private isTensorOption (t: System.Type) =
        t.IsGenericType
        && t.GetGenericTypeDefinition() = typedefof<option<_>>
        && t.GetGenericArguments()[0] = tensorType

    let private isRecordWithTensors (t: System.Type) =
        FSharpType.IsRecord(t, BindingFlags.Public ||| BindingFlags.Instance)

    let rec private collectTensors (prefix: string) (value: obj) (ty: System.Type) : (string * Tensor) list =
        if not (FSharpType.IsRecord(ty, BindingFlags.Public ||| BindingFlags.Instance)) then
            []
        else
            let fields =
                FSharpType.GetRecordFields(ty, BindingFlags.Public ||| BindingFlags.Instance)

            fields
            |> Array.toList
            |> List.collect (fun fi ->
                let fieldVal = fi.GetValue value

                let path =
                    if prefix.Length = 0 then
                        fi.Name
                    else
                        prefix + "." + fi.Name

                if fi.PropertyType = tensorType then
                    [ path, fieldVal :?> Tensor ]
                elif isTensorOption fi.PropertyType then
                    match fieldVal :?> Tensor option with
                    | Some t -> [ path, t ]
                    | None -> []
                elif isRecordWithTensors fi.PropertyType then
                    collectTensors path fieldVal fi.PropertyType
                else
                    [])

    let namedParams (model: 'T) : (string * Tensor) list =
        collectTensors "" (box model) typeof<'T>

    let trainableVars (model: 'T) : Tensor list =
        namedParams model
        |> List.choose (fun (_, t) -> if t.RequiresGrad then Some t else None)

    let save (model: 'T) (dirPath: string) : Result<unit, ToroError> =
        result {
            do! ToroError.wrap (fun () -> Directory.CreateDirectory dirPath |> ignore)

            for name, tensor in namedParams model do
                let filePath = Path.Combine(dirPath, name + ".toro")
                let dir = Path.GetDirectoryName filePath

                if not (Directory.Exists dir) then
                    Directory.CreateDirectory dir |> ignore

                do! tensor.save filePath
        }

    let loadInto (model: 'T) (dirPath: string) : Result<unit, ToroError> =
        result {
            for name, tensor in namedParams model do
                let filePath = Path.Combine(dirPath, name + ".toro")

                if File.Exists filePath then
                    let! loaded = Tensor.load filePath
                    do! tensor.copyInPlace loaded
        }
