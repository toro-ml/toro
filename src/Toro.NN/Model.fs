namespace Toro.NN

open System.IO
open System.Reflection
open Microsoft.FSharp.Reflection
open Toro

module Model =

    let private flags = BindingFlags.Public ||| BindingFlags.Instance

    let private isTensorOption (t: System.Type) =
        t.IsGenericType
        && t.GetGenericTypeDefinition() = typedefof<option<_>>
        && t.GetGenericArguments()[0] = typeof<Tensor>

    let private makePath prefix name =
        if prefix = "" then name else $"{prefix}.{name}"

    let rec private collect (prefix: string) (value: obj) : (string * Tensor) list =
        match value with
        | null -> []
        | :? Tensor as t -> [ prefix, t ]
        | _ ->
            let ty = value.GetType()

            if FSharpType.IsRecord(ty, flags) then
                collectRecord prefix value ty
            elif
                typeof<System.Collections.IEnumerable>.IsAssignableFrom ty
                && ty <> typeof<string>
            then
                collectSeq prefix (value :?> System.Collections.IEnumerable)
            else
                []

    and private collectRecord (prefix: string) (value: obj) (ty: System.Type) : (string * Tensor) list =
        FSharpType.GetRecordFields(ty, flags)
        |> Array.toList
        |> List.collect (fun fi ->
            let path = makePath prefix fi.Name

            if isTensorOption fi.PropertyType then
                match fi.GetValue value :?> Tensor option with
                | Some t -> [ path, t ]
                | None -> []
            else
                collect path (fi.GetValue value))

    and private collectSeq (prefix: string) (items: System.Collections.IEnumerable) : (string * Tensor) list =
        items
        |> Seq.cast<obj>
        |> Seq.indexed
        |> Seq.collect (fun (i, item) -> collect (makePath prefix (string i)) item)
        |> Seq.toList

    let namedParams (model: 'T) : (string * Tensor) list = collect "" (box model)

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
