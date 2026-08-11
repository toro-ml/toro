namespace Toro.NN

open System.IO
open System.Reflection
open Microsoft.FSharp.Reflection
open TorchSharp
open Toro

/// Describes a shape or dtype mismatch between model and loaded tensor.
type TensorMismatch = {
    Name: string
    Expected: string
    Got: string
}

/// Report produced after loading tensors into a model.
type LoadReport = {
    Loaded: string list
    Missing: string list
    Unexpected: string list
    ShapeMismatches: TensorMismatch list
    DTypeMismatches: TensorMismatch list
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

    let private validateUniqueNames (ps: (string * Tensor) list) : unit =
        let duplicates =
            ps
            |> List.countBy fst
            |> List.filter (fun (_, c) -> c > 1)
            |> List.map fst

        match duplicates with
        | [] -> ()
        | dupes -> failwith $"Duplicate parameter names: %A{dupes}"

    let private formatShape (shape: int64[]) =
        sprintf "[%s]" (shape |> Array.map string |> String.concat ", ")

    type private ParamMatch<'Source> =
        | Matched of name: string * target: Tensor * source: 'Source
        | MissingParam of name: string
        | ShapeMismatch of TensorMismatch
        | DTypeMismatch of TensorMismatch

    let private classifyParam
        (lookup: Map<string, 'Source>)
        (getShape: 'Source -> int64[])
        (getDType: 'Source -> torch.ScalarType)
        (name: string)
        (tensor: Tensor)
        : ParamMatch<'Source> =
        match Map.tryFind name lookup with
        | None -> MissingParam name
        | Some src ->
            let srcShape = getShape src
            let srcDType = getDType src

            if tensor.shape <> srcShape then
                ShapeMismatch {
                    Name = name
                    Expected = formatShape tensor.shape
                    Got = formatShape srcShape
                }
            elif tensor.dtype <> srcDType then
                DTypeMismatch {
                    Name = name
                    Expected = string tensor.dtype
                    Got = string srcDType
                }
            else
                Matched(name, tensor, src)

    let private buildReport (matches: ParamMatch<'Source> list) (unexpected: string list) : LoadReport = {
        Loaded =
            matches
            |> List.choose (function
                | Matched(n, _, _) -> Some n
                | _ -> None)
        Missing =
            matches
            |> List.choose (function
                | MissingParam n -> Some n
                | _ -> None)
        Unexpected = unexpected
        ShapeMismatches =
            matches
            |> List.choose (function
                | ShapeMismatch mm -> Some mm
                | _ -> None)
        DTypeMismatches =
            matches
            |> List.choose (function
                | DTypeMismatch mm -> Some mm
                | _ -> None)
    }

    let private enforceStrict (report: LoadReport) (mode: LoadMode) : unit =
        match mode with
        | Strict when
            report.Missing <> []
            || report.Unexpected <> []
            || report.ShapeMismatches <> []
            || report.DTypeMismatches <> []
            ->
            let parts = [
                if report.Missing <> [] then
                    $"missing keys: %A{report.Missing}"
                if report.Unexpected <> [] then
                    $"unexpected keys: %A{report.Unexpected}"
                if report.ShapeMismatches <> [] then
                    let names = report.ShapeMismatches |> List.map _.Name
                    $"shape mismatches: %A{names}"
                if report.DTypeMismatches <> [] then
                    let names = report.DTypeMismatches |> List.map _.Name
                    $"dtype mismatches: %A{names}"
            ]

            failwith (parts |> String.concat "; ")
        | _ -> ()

    /// Return all named tensors in the model via reflection.
    let namedParams (model: 'T) : (string * Tensor) list = collect "" (box model)

    /// Return all tensors that require gradients.
    let trainableVars (model: 'T) : Tensor list =
        namedParams model
        |> List.choose (fun (_, t) -> if t.requires_grad then Some t else None)

    /// Save all model tensors to a .safetensors file.
    let save (model: 'T) (filePath: string) : unit =
        let ps = namedParams model
        validateUniqueNames ps
        let tensors = ps |> Map.ofList
        SafeTensors.save tensors filePath

    /// Load tensors from a dictionary into the model, matching by parameter name.
    /// When nameMap is Some, dictionary keys are translated before matching.
    let loadFromDict
        (model: 'T)
        (tensors: Map<string, Tensor>)
        (nameMap: Map<string, string> option)
        (mode: LoadMode)
        : LoadReport =
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

        let matches =
            modelNames
            |> List.map (fun (name, tensor) -> classifyParam lookup (_.shape) (_.dtype) name tensor)

        let modelNameSet = modelNames |> List.map fst |> Set.ofList

        let unexpected =
            lookup
            |> Map.toList
            |> List.map fst
            |> List.filter (fun k -> not (Set.contains k modelNameSet))

        let report = buildReport matches unexpected
        enforceStrict report mode

        for m in matches do
            match m with
            | Matched(_, target, src) -> target.copyInPlace src
            | _ -> ()

        report

    /// Load tensors from a .safetensors file into the model in place.
    /// Shape and dtype are validated from the header before any tensor data is read.
    /// Only the tensors that match both shape and dtype are loaded into memory.
    let loadInto (model: 'T) (filePath: string) (mode: LoadMode) : LoadReport =
        let modelNames = namedParams model
        let neededNames = modelNames |> List.map fst |> Set.ofList

        let allMeta = SafeTensors.loadMeta filePath
        let allFileKeys = allMeta |> Map.toList |> List.map fst

        let unexpected =
            allFileKeys
            |> List.filter (fun k -> not (Set.contains k neededNames))

        let matches =
            modelNames
            |> List.map (fun (name, tensor) -> classifyParam allMeta (fun m -> m.Shape) (fun m -> m.DType) name tensor)

        let report = buildReport matches unexpected
        enforceStrict report mode

        let namesToLoad =
            matches
            |> List.choose (function
                | Matched(n, _, _) -> Some n
                | _ -> None)
            |> Set.ofList

        scoped {
            let _, tensors = SafeTensors.loadSelected filePath namesToLoad

            for m in matches do
                match m with
                | Matched(name, target, _) ->
                    match Map.tryFind name tensors with
                    | Some src -> target.copyInPlace src
                    | None -> ()
                | _ -> ()
        }

        report
