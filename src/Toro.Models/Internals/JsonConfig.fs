namespace Toro.Models

open System.Text.Json

module internal JsonConfig =

    let validateObject (label: string) (root: JsonElement) =
        if root.ValueKind <> JsonValueKind.Object then
            invalidOp $"{label} config must be a JSON object."

        root.EnumerateObject()
        |> Seq.countBy _.Name
        |> Seq.tryFind (fun (_, count) -> count > 1)
        |> Option.iter (fun (name, _) -> invalidOp $"{label} config contains duplicate key '{name}'.")

    let tryProperty (root: JsonElement) (name: string) =
        match root.TryGetProperty name with
        | true, value -> Some value
        | false, _ -> None

    let property (label: string) (root: JsonElement) (name: string) =
        tryProperty root name
        |> Option.defaultWith (fun () -> invalidOp $"{label} config is missing '{name}'.")

    let int64Element (label: string) (name: string) (value: JsonElement) =
        match value.TryGetInt64() with
        | true, result -> result
        | false, _ -> invalidOp $"{label} config '{name}' must be an integer."

    let int64Value label (root: JsonElement) name =
        property label root name |> int64Element label name

    let floatValue label (root: JsonElement) name =
        let value = property label root name

        if value.ValueKind <> JsonValueKind.Number then
            invalidOp $"{label} config '{name}' must be a number."

        value.GetDouble()

    let boolValue label (root: JsonElement) name =
        match (property label root name).ValueKind with
        | JsonValueKind.True -> true
        | JsonValueKind.False -> false
        | _ -> invalidOp $"{label} config '{name}' must be a boolean."

    let stringValue label (root: JsonElement) name =
        let value = property label root name

        if value.ValueKind <> JsonValueKind.String then
            invalidOp $"{label} config '{name}' must be a string."

        value.GetString()
