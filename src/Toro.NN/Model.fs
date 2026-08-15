namespace Toro.NN

open System
open System.Collections
open System.Collections.Concurrent
open System.Collections.Generic
open System.Reflection
open Microsoft.FSharp.Reflection
open TorchSharp
open Toro

/// Marks a record field, or every tensor below a container field, as parameters.
[<Sealed; AttributeUsage(AttributeTargets.Property)>]
type ParameterAttribute() =
    inherit Attribute()

/// Marks a record field, or every tensor below a container field, as persistent buffers.
[<Sealed; AttributeUsage(AttributeTargets.Property)>]
type BufferAttribute() =
    inherit Attribute()

/// Excludes a record field and all values below it from model state discovery.
[<Sealed; AttributeUsage(AttributeTargets.Property)>]
type ModelIgnoreAttribute() =
    inherit Attribute()

/// Classifies a tensor stored in model state.
type TensorKind =
    | Parameter
    | Buffer

/// A canonical, named tensor in model state.
type NamedTensor = {
    Name: string
    Tensor: Tensor
    Kind: TensorKind
}

/// A validated snapshot of canonical model-state names and tensor identities.
type ModelState = private ModelState of NamedTensor list

/// Explicitly describes model parameters, buffers, and resource disposal without reflection.
type ModelDescriptor<'Model> = {
    NamedParameters: 'Model -> seq<string * Tensor>
    NamedBuffers: 'Model -> seq<string * Tensor>
    Dispose: 'Model -> unit
}

/// Describes a shape or dtype mismatch between model and loaded tensor.
type TensorMismatch = {
    Name: string
    Expected: string
    Got: string
}

/// Report produced after loading tensors into a model.
type LoadReport = {
    Loaded: string list
    Ignored: string list
    Missing: string list
    Unexpected: string list
    ShapeMismatches: TensorMismatch list
    DTypeMismatches: TensorMismatch list
}

/// Controls whether missing or unexpected keys cause an error.
type LoadMode =
    | Strict
    | Lenient

type private StateMatch<'Source> =
    | Matched of state: NamedTensor * source: 'Source
    | MissingState of name: string
    | ShapeMismatch of TensorMismatch
    | DTypeMismatch of TensorMismatch

/// Operations on explicit model descriptors.
module ModelDescriptor =

    /// Dispose resources owned by a model using its explicit descriptor.
    let dispose (descriptor: ModelDescriptor<'Model>) (model: 'Model) = descriptor.Dispose model

/// Attribute-driven model state discovery.
/// Records use field names; options are transparent; tuples and stable lists use zero-based
/// indices; union values use CaseName.index; and string dictionaries use ordinally sorted keys.
/// Supported containers are records, options, tuples, discriminated unions, arrays, F# lists,
/// ResizeArray/IList/IReadOnlyList, and string-keyed dictionaries.
module Model =

    let private flags = BindingFlags.Public ||| BindingFlags.Instance

    type private FieldDirective =
        | Descend
        | AsParameter
        | AsBuffer
        | Ignore
        | Conflicting

    type private FieldPlan = {
        Property: PropertyInfo
        Directive: FieldDirective
    }

    type private TypePlan =
        | Record
        | Tuple
        | Option
        | Dictionary of keyType: Type option
        | Sequence
        | Union
        | Scalar
        | UnsupportedEnumerable

    let private typePlans = ConcurrentDictionary<Type, TypePlan>()
    let private recordPlans = ConcurrentDictionary<Type, FieldPlan[]>()

    let private displayPath prefix =
        if String.IsNullOrEmpty prefix then "<root>" else prefix

    let private makePath prefix name =
        if String.IsNullOrEmpty prefix then
            name
        else
            $"{prefix}.{name}"

    let private isGenericDefinition (definition: Type) (candidate: Type) =
        candidate.IsGenericType
        && candidate.GetGenericTypeDefinition() = definition

    let private tryGenericInterface (definitions: Type list) (ty: Type) =
        Array.append [| ty |] (ty.GetInterfaces())
        |> Array.tryFind (fun candidate ->
            candidate.IsGenericType
            && definitions
               |> List.contains (candidate.GetGenericTypeDefinition()))

    let private classifyType (ty: Type) =
        let dictionaryInterface =
            tryGenericInterface [ typedefof<IDictionary<_, _>>; typedefof<IReadOnlyDictionary<_, _>> ] ty

        let isList = isGenericDefinition typedefof<list<_>> ty

        let isReadOnlyList =
            tryGenericInterface [ typedefof<IReadOnlyList<_>> ] ty
            |> Option.isSome

        match dictionaryInterface with
        | Some interfaceType -> Dictionary(Some(interfaceType.GetGenericArguments()[0]))
        | None when typeof<IDictionary>.IsAssignableFrom ty -> Dictionary None
        | None when FSharpType.IsRecord(ty, flags) -> Record
        | None when FSharpType.IsTuple ty -> Tuple
        | None when isGenericDefinition typedefof<option<_>> ty -> Option
        | None when ty.IsArray || isList || isReadOnlyList -> Sequence
        | None when FSharpType.IsUnion(ty, flags) -> Union
        | None when
            ty <> typeof<string>
            && typeof<IEnumerable>.IsAssignableFrom ty
            ->
            UnsupportedEnumerable
        | None -> Scalar

    let private getTypePlan (ty: Type) = typePlans.GetOrAdd(ty, classifyType)

    let private fieldDirective (property: PropertyInfo) =
        let parameter = property.IsDefined(typeof<ParameterAttribute>, true)
        let buffer = property.IsDefined(typeof<BufferAttribute>, true)
        let ignore = property.IsDefined(typeof<ModelIgnoreAttribute>, true)

        match
            [ parameter; buffer; ignore ]
            |> List.filter id
            |> List.length
        with
        | 0 -> Descend
        | 1 when parameter -> AsParameter
        | 1 when buffer -> AsBuffer
        | 1 -> Ignore
        | _ -> Conflicting

    let private buildRecordPlan (ty: Type) =
        FSharpType.GetRecordFields(ty, flags)
        |> Array.map (fun property -> {
            Property = property
            Directive = fieldDirective property
        })

    let private getRecordPlan (ty: Type) =
        recordPlans.GetOrAdd(ty, buildRecordPlan)

    let private isDirectTensorType (ty: Type) =
        ty = typeof<Tensor>
        || (isGenericDefinition typedefof<option<_>> ty
            && ty.GetGenericArguments()[0] = typeof<Tensor>)

    let private mergeKind path inherited declared =
        match inherited, declared with
        | None, kind -> Some kind
        | Some current, kind when current = kind -> inherited
        | Some current, kind -> invalidOp $"Model state kind conflict at '{path}': inherited {current}, declared {kind}."

    let private validateDictionaryKey prefix (key: string) =
        let path = displayPath prefix

        if String.IsNullOrEmpty key then
            invalidOp $"Dictionary at '{path}' contains an empty key."

        if key.Contains('.') then
            invalidOp $"Dictionary key '{key}' at '{path}' cannot contain '.'."

    let private withCycleCheck
        (ancestors: HashSet<obj>)
        prefix
        (value: obj)
        (collectValue: unit -> (string * Tensor * TensorKind) list)
        =
        if value.GetType().IsValueType then
            collectValue ()
        elif ancestors.Add value then
            try
                collectValue ()
            finally
                ancestors.Remove value |> ignore
        else
            invalidOp $"Cycle detected while collecting model state at '{displayPath prefix}'."

    let rec private collect
        (ancestors: HashSet<obj>)
        (kind: TensorKind option)
        prefix
        (value: obj)
        : (string * Tensor * TensorKind) list =
        match value with
        | null -> []
        | :? Tensor as tensor ->
            match kind with
            | Some tensorKind -> [ prefix, tensor, tensorKind ]
            | None -> invalidOp $"Tensor at '{displayPath prefix}' is not marked as Parameter or Buffer."
        | _ ->
            let ty = value.GetType()

            match getTypePlan ty with
            | Record -> withCycleCheck ancestors prefix value (fun () -> collectRecord ancestors kind prefix value ty)
            | Tuple -> withCycleCheck ancestors prefix value (fun () -> collectTuple ancestors kind prefix value)
            | Option -> withCycleCheck ancestors prefix value (fun () -> collectOption ancestors kind prefix value ty)
            | Dictionary keyType ->
                match keyType with
                | Some actual when actual <> typeof<string> ->
                    invalidOp $"Dictionary at '{displayPath prefix}' must use string keys, but uses {actual.FullName}."
                | _ ->
                    withCycleCheck ancestors prefix value (fun () ->
                        collectDictionary ancestors kind prefix (value :?> IEnumerable))
            | Sequence ->
                withCycleCheck ancestors prefix value (fun () -> collectSequence ancestors kind prefix (value :?> IEnumerable))
            | Union -> withCycleCheck ancestors prefix value (fun () -> collectUnion ancestors kind prefix value ty)
            | UnsupportedEnumerable ->
                invalidOp
                    $"Enumerable type {ty.FullName} at '{displayPath prefix}' is not a stable model collection. Materialize it as an array, F# list, ResizeArray, or IReadOnlyList."
            | Scalar -> []

    and private collectRecord ancestors inherited prefix value ty =
        getRecordPlan ty
        |> Array.toList
        |> List.collect (fun field ->
            let path = makePath prefix field.Property.Name

            match field.Directive with
            | Ignore -> []
            | Conflicting -> invalidOp $"Field '{path}' has conflicting model state attributes."
            | Descend ->
                if
                    inherited.IsNone
                    && isDirectTensorType field.Property.PropertyType
                then
                    invalidOp $"Tensor field '{path}' is not marked as Parameter or Buffer."

                collect ancestors inherited path (field.Property.GetValue value)
            | AsParameter ->
                let nextKind = mergeKind path inherited Parameter
                collect ancestors nextKind path (field.Property.GetValue value)
            | AsBuffer ->
                let nextKind = mergeKind path inherited Buffer
                collect ancestors nextKind path (field.Property.GetValue value))

    and private collectTuple ancestors kind prefix value =
        FSharpValue.GetTupleFields value
        |> Array.indexed
        |> Array.toList
        |> List.collect (fun (index, field) -> collect ancestors kind (makePath prefix (string index)) field)

    and private collectOption ancestors kind prefix value ty =
        let _, fields = FSharpValue.GetUnionFields(value, ty, flags)

        fields
        |> Array.toList
        |> List.collect (collect ancestors kind prefix)

    and private collectUnion ancestors kind prefix value ty =
        let caseInfo, fields = FSharpValue.GetUnionFields(value, ty, flags)
        let casePath = makePath prefix caseInfo.Name

        fields
        |> Array.indexed
        |> Array.toList
        |> List.collect (fun (index, field) -> collect ancestors kind (makePath casePath (string index)) field)

    and private collectSequence ancestors kind prefix (items: IEnumerable) =
        items
        |> Seq.cast<obj>
        |> Seq.indexed
        |> Seq.collect (fun (index, item) -> collect ancestors kind (makePath prefix (string index)) item)
        |> Seq.toList

    and private collectDictionary ancestors kind prefix (items: IEnumerable) =
        items
        |> Seq.cast<obj>
        |> Seq.map (fun item ->
            let itemType = item.GetType()
            let keyProperty = itemType.GetProperty("Key", flags)
            let valueProperty = itemType.GetProperty("Value", flags)

            if isNull keyProperty || isNull valueProperty then
                invalidOp $"Dictionary at '{displayPath prefix}' returned unsupported entry type {itemType.FullName}."

            let keyValue = keyProperty.GetValue item

            let key =
                match keyValue with
                | :? string as key -> key
                | null -> invalidOp $"Dictionary at '{displayPath prefix}' contains a null key."
                | _ ->
                    invalidOp
                        $"Dictionary at '{displayPath prefix}' must use string keys, but uses {keyValue.GetType().FullName}."

            validateDictionaryKey prefix key
            key, valueProperty.GetValue item)
        |> Seq.sortWith (fun (left, _) (right, _) -> StringComparer.Ordinal.Compare(left, right))
        |> Seq.collect (fun (key, item) -> collect ancestors kind (makePath prefix key) item)
        |> Seq.toList

    let private canonicalize (entries: (string * Tensor * TensorKind) list) =
        let names = HashSet<string>(StringComparer.Ordinal)

        let tensors =
            System.Collections.Generic.Dictionary<obj, NamedTensor>(ReferenceEqualityComparer.Instance)

        let canonical = ResizeArray<NamedTensor>()

        for name, tensor, kind in entries do
            if String.IsNullOrWhiteSpace name then
                invalidOp "Model state contains a tensor without a name."

            if
                name.Split('.', StringSplitOptions.None)
                |> Array.exists String.IsNullOrEmpty
            then
                invalidOp $"Model state name '{name}' contains an empty path segment."

            if isNull (box tensor) then
                invalidOp $"Model state '{name}' contains a null Tensor."

            if not (names.Add name) then
                invalidOp $"Duplicate model state name: '{name}'."

            let key = box tensor

            match tensors.TryGetValue key with
            | true, existing when existing.Kind <> kind ->
                invalidOp $"Shared tensor has conflicting roles at '{existing.Name}' ({existing.Kind}) and '{name}' ({kind})."
            | true, _ -> ()
            | false, _ ->
                let item = {
                    Name = name
                    Tensor = tensor
                    Kind = kind
                }

                tensors.Add(key, item)
                canonical.Add item

        canonical |> Seq.toList

    /// Discover model state through attributes and supported F# containers. A shared Tensor is
    /// represented once, under the first path at which it is discovered.
    let state (model: 'Model) : ModelState =
        let ancestors = HashSet<obj>(ReferenceEqualityComparer.Instance)

        collect ancestors None "" (box model)
        |> canonicalize
        |> ModelState

    /// Build model state from an explicit descriptor without inspecting the model structure.
    let stateWith (descriptor: ModelDescriptor<'Model>) (model: 'Model) : ModelState =
        let parameters =
            descriptor.NamedParameters model
            |> Seq.map (fun (name, tensor) -> name, tensor, Parameter)

        let buffers =
            descriptor.NamedBuffers model
            |> Seq.map (fun (name, tensor) -> name, tensor, Buffer)

        Seq.append parameters buffers
        |> Seq.toList
        |> canonicalize
        |> ModelState

/// Query, save, and load a validated model-state view.
module ModelState =

    /// Return parameters and buffers in deterministic canonical order.
    let namedState (ModelState state) : NamedTensor list = state

    /// Return canonical parameters, including parameters that do not require gradients.
    let namedParams state : NamedTensor list =
        namedState state
        |> List.filter (fun item -> item.Kind = Parameter)

    /// Return canonical persistent buffers in deterministic order.
    let namedBuffers state : NamedTensor list =
        namedState state
        |> List.filter (fun item -> item.Kind = Buffer)

    /// Return canonical parameters whose Tensor currently requires gradients.
    let trainableParams state : NamedTensor list =
        namedParams state
        |> List.filter (fun item -> item.Tensor.requires_grad)

    let private formatShape (shape: int64[]) =
        sprintf "[%s]" (shape |> Array.map string |> String.concat ", ")

    let private classifyState
        (lookup: Map<string, 'Source>)
        (getShape: 'Source -> int64[])
        (getDType: 'Source -> torch.ScalarType)
        (state: NamedTensor)
        =
        match Map.tryFind state.Name lookup with
        | None -> MissingState state.Name
        | Some source ->
            let sourceShape = getShape source
            let sourceDType = getDType source

            if state.Tensor.shape <> sourceShape then
                ShapeMismatch {
                    Name = state.Name
                    Expected = formatShape state.Tensor.shape
                    Got = formatShape sourceShape
                }
            elif state.Tensor.dtype <> sourceDType then
                DTypeMismatch {
                    Name = state.Name
                    Expected = string state.Tensor.dtype
                    Got = string sourceDType
                }
            else
                Matched(state, source)

    let private buildReport matches ignored unexpected = {
        Loaded =
            matches
            |> List.choose (function
                | Matched(state, _) -> Some state.Name
                | _ -> None)
        Ignored = ignored
        Missing =
            matches
            |> List.choose (function
                | MissingState name -> Some name
                | _ -> None)
        Unexpected = unexpected
        ShapeMismatches =
            matches
            |> List.choose (function
                | ShapeMismatch mismatch -> Some mismatch
                | _ -> None)
        DTypeMismatches =
            matches
            |> List.choose (function
                | DTypeMismatch mismatch -> Some mismatch
                | _ -> None)
    }

    let private enforceStrict report mode =
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
                    $"shape mismatches: %A{report.ShapeMismatches |> List.map _.Name}"
                if report.DTypeMismatches <> [] then
                    $"dtype mismatches: %A{report.DTypeMismatches |> List.map _.Name}"
            ]

            invalidOp (String.concat "; " parts)
        | _ -> ()

    let private mappedLookup tensors mapping =
        let mapped, ignored =
            tensors
            |> Map.toList
            |> List.fold
                (fun (mapped, ignored) (sourceName, tensor) ->
                    match NameMapping.resolve mapping sourceName with
                    | NameResolution.Keep -> (sourceName, sourceName, tensor) :: mapped, ignored
                    | NameResolution.Rename targetName -> (sourceName, targetName, tensor) :: mapped, ignored
                    | NameResolution.Ignore -> mapped, sourceName :: ignored)
                ([], [])

        let mapped = List.rev mapped
        let ignored = List.rev ignored

        let collisions =
            mapped
            |> List.groupBy (fun (_, targetName, _) -> targetName)
            |> List.choose (fun (targetName, entries) ->
                match entries with
                | [ _ ] -> None
                | _ -> Some(targetName, entries |> List.map (fun (sourceName, _, _) -> sourceName)))

        match collisions with
        | [] ->
            mapped
            |> List.map (fun (_, targetName, tensor) -> targetName, tensor)
            |> Map.ofList,
            ignored
        | (targetName, sourceNames) :: _ ->
            invalidOp $"Name mapping maps multiple source keys %A{sourceNames} to '{targetName}'."

    let private planLoad lookup ignored getShape getDType modelState mode =
        let entries = namedState modelState
        let matches = entries |> List.map (classifyState lookup getShape getDType)
        let stateNames = entries |> List.map _.Name |> Set.ofList

        let unexpected =
            lookup
            |> Map.toList
            |> List.map fst
            |> List.filter (fun name -> not (Set.contains name stateNames))

        let report = buildReport matches ignored unexpected
        enforceStrict report mode
        report, matches

    let internal prepareLoadFromDict mapping mode modelState tensors =
        let lookup, ignored = mappedLookup tensors mapping

        let report, matches =
            planLoad
                lookup
                ignored
                (fun (tensor: Tensor) -> tensor.shape)
                (fun (tensor: Tensor) -> tensor.dtype)
                modelState
                mode

        let commit () =
            for matched in matches do
                match matched with
                | Matched(state, source) -> state.Tensor.copyInPlace source
                | _ -> ()

        report, commit

    let internal prepareLoadSafeTensors mapping mode modelState (reader: SafeTensorReader) =
        let sources =
            reader.Metadata
            |> Map.map (fun sourceName metadata -> sourceName, metadata)

        let lookup, ignored = mappedLookup sources mapping

        let report, matches =
            planLoad
                lookup
                ignored
                (fun (_, metadata: TensorMeta) -> metadata.Shape)
                (fun (_, metadata: TensorMeta) -> metadata.DType)
                modelState
                mode

        let commit () =
            for matched in matches do
                match matched with
                | Matched(state, (sourceName, _)) ->
                    use source = reader.ReadTensor sourceName
                    state.Tensor.copyInPlace source
                | _ -> ()

        report, commit

    /// Save canonical parameters and buffers to a .safetensors file.
    let save modelState (filePath: string) : unit =
        namedState modelState
        |> List.map (fun item -> item.Name, item.Tensor)
        |> Map.ofList
        |> fun tensors -> SafeTensors.save tensors filePath

    /// Load tensors from a dictionary into the model, matching by canonical state name.
    /// Validation completes before any model tensor is changed.
    let loadFromDict (mode: LoadMode) modelState (tensors: Map<string, Tensor>) =
        let report, commit =
            prepareLoadFromDict NameMapping.identity mode modelState tensors

        commit ()
        report

    /// Translate or ignore external tensor names declaratively, then load matching canonical state.
    /// Mapping, collision, shape, and dtype validation completes before any model tensor is changed.
    let loadFromDictWith (mapping: NameMapping) (mode: LoadMode) modelState (tensors: Map<string, Tensor>) =
        let report, commit = prepareLoadFromDict mapping mode modelState tensors
        commit ()
        report

    /// Load tensors from a validated SafeTensors reader into model state.
    /// Validation completes before any model tensor is changed.
    let loadSafeTensors (mode: LoadMode) modelState (reader: SafeTensorReader) : LoadReport =
        let report, commit =
            prepareLoadSafeTensors NameMapping.identity mode modelState reader

        commit ()
        report

    /// Translate or ignore external names, then load one validated SafeTensors entry at a time.
    /// Mapping, name, shape, and dtype validation completes before any tensor is changed.
    let loadSafeTensorsWith (mapping: NameMapping) (mode: LoadMode) modelState (reader: SafeTensorReader) : LoadReport =
        let report, commit = prepareLoadSafeTensors mapping mode modelState reader
        commit ()
        report
