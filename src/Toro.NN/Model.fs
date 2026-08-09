namespace Toro.NN

open System.IO
open System.Reflection
open Microsoft.FSharp.Reflection
open Toro

/// Report produced after loading tensors into a model.
type LoadReport = {
    Loaded: string list
    Missing: string list
    Unexpected: string list
}

/// Controls whether missing or unexpected keys cause an error.
type LoadMode =
    | Strict
    | Lenient

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

    let private validateUniqueNames (ps: (string * Tensor) list) : Result<unit, ToroError> =
        let duplicates =
            ps
            |> List.countBy fst
            |> List.filter (fun (_, c) -> c > 1)
            |> List.map fst

        match duplicates with
        | [] -> Ok()
        | dupes -> Error(Msg $"Duplicate parameter names: %A{dupes}")

    /// Return all named tensors in the model via reflection.
    let namedParams (model: 'T) : (string * Tensor) list = collect "" (box model)

    /// Return all tensors that require gradients.
    let trainableVars (model: 'T) : Tensor list =
        namedParams model
        |> List.choose (fun (_, t) -> if t.RequiresGrad then Some t else None)

    /// Save all model tensors to a .safetensors file.
    let save (model: 'T) (filePath: string) : Result<unit, ToroError> =
        result {
            let ps = namedParams model
            do! validateUniqueNames ps
            let tensors = ps |> Map.ofList
            do! SafeTensors.save tensors filePath
        }

    /// Load tensors from a dictionary into the model, matching by parameter name.
    /// When nameMap is Some, dictionary keys are translated before matching.
    let loadFromDict
        (model: 'T)
        (tensors: Map<string, Tensor>)
        (nameMap: Map<string, string> option)
        (mode: LoadMode)
        : Result<LoadReport, ToroError> =
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

            let modelNames = namedParams model
            let mutable loaded = []
            let mutable missing = []

            for name, tensor in modelNames do
                match lookup |> Map.tryFind name with
                | Some src ->
                    do! tensor.copyInPlace src
                    loaded <- name :: loaded
                | None -> missing <- name :: missing

            let modelNameSet = modelNames |> List.map fst |> Set.ofList

            let unexpected =
                lookup
                |> Map.toList
                |> List.map fst
                |> List.filter (fun k -> not (Set.contains k modelNameSet))

            let report = {
                Loaded = List.rev loaded
                Missing = List.rev missing
                Unexpected = unexpected
            }

            match mode with
            | Strict when report.Missing <> [] || report.Unexpected <> [] ->
                let parts = [
                    if report.Missing <> [] then
                        $"missing keys: %A{report.Missing}"
                    if report.Unexpected <> [] then
                        $"unexpected keys: %A{report.Unexpected}"
                ]

                return! Error(Msg(parts |> String.concat "; "))
            | _ -> return report
        }

    /// Load tensors from a .safetensors file into the model in place.
    let loadInto (model: 'T) (filePath: string) (mode: LoadMode) : Result<LoadReport, ToroError> =
        result {
            let! tensors = SafeTensors.load filePath
            return! loadFromDict model tensors None mode
        }
