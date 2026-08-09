namespace Toro.NN

open System.IO
open System.Reflection
open Microsoft.FSharp.Reflection
open Toro

/// Reflection-based parameter discovery, serialization, and deserialization for F# record models.
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

    /// Return all named tensors in the model via reflection.
    let namedParams (model: 'T) : (string * Tensor) list = collect "" (box model)

    /// Return all tensors that require gradients.
    let trainableVars (model: 'T) : Tensor list =
        namedParams model
        |> List.choose (fun (_, t) -> if t.RequiresGrad then Some t else None)

    /// Save all model tensors to the given directory.
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

    /// Load tensors from the given directory into the model in place.
    let loadInto (model: 'T) (dirPath: string) : Result<unit, ToroError> =
        result {
            for name, tensor in namedParams model do
                let filePath = Path.Combine(dirPath, name + ".toro")

                if File.Exists filePath then
                    let! loaded = Tensor.load filePath
                    do! tensor.copyInPlace loaded
        }

    /// Load tensors from a dictionary into the model, matching by parameter name.
    /// When nameMap is Some, dictionary keys are translated before matching.
    let loadFromDict
        (model: 'T)
        (tensors: Map<string, Tensor>)
        (nameMap: Map<string, string> option)
        : Result<unit, ToroError> =
        result {
            let lookup =
                match nameMap with
                | Some mapping ->
                    tensors
                    |> Map.toSeq
                    |> Seq.map (fun (k, v) ->
                        let mapped = mapping |> Map.tryFind k |> Option.defaultValue k
                        mapped, v)
                    |> Map.ofSeq
                | None -> tensors

            for name, tensor in namedParams model do
                match lookup |> Map.tryFind name with
                | Some src -> do! tensor.copyInPlace src
                | None -> ()
        }
