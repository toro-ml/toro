namespace Toro.Models.Interop

open System.ComponentModel
open System.Text.Json

/// JSON configuration helpers for Toro model-family packages.
[<EditorBrowsable(EditorBrowsableState.Never)>]
module JsonConfig =

    /// Validate that a configuration root is an object without duplicate keys.
    let validateObject (label: string) (root: JsonElement) =
        if root.ValueKind <> JsonValueKind.Object then
            invalidOp $"{label} config must be a JSON object."

        root.EnumerateObject()
        |> Seq.countBy _.Name
        |> Seq.tryFind (fun (_, count) -> count > 1)
        |> Option.iter (fun (name, _) -> invalidOp $"{label} config contains duplicate key '{name}'.")

    /// Find an optional object property.
    let tryProperty (root: JsonElement) (name: string) =
        match root.TryGetProperty name with
        | true, value -> Some value
        | false, _ -> None

    /// Get a required object property.
    let property (label: string) (root: JsonElement) (name: string) =
        tryProperty root name
        |> Option.defaultWith (fun () -> invalidOp $"{label} config is missing '{name}'.")

    /// Read an integer JSON element.
    let int64Element (label: string) (name: string) (value: JsonElement) =
        match value.TryGetInt64() with
        | true, result -> result
        | false, _ -> invalidOp $"{label} config '{name}' must be an integer."

    /// Read a required integer property.
    let int64Value label (root: JsonElement) name =
        property label root name |> int64Element label name

    /// Read a required floating-point property.
    let floatValue label (root: JsonElement) name =
        let value = property label root name

        if value.ValueKind <> JsonValueKind.Number then
            invalidOp $"{label} config '{name}' must be a number."

        value.GetDouble()

    /// Read a required Boolean property.
    let boolValue label (root: JsonElement) name =
        match (property label root name).ValueKind with
        | JsonValueKind.True -> true
        | JsonValueKind.False -> false
        | _ -> invalidOp $"{label} config '{name}' must be a boolean."

    /// Read a required string property.
    let stringValue label (root: JsonElement) name =
        let value = property label root name

        if value.ValueKind <> JsonValueKind.String then
            invalidOp $"{label} config '{name}' must be a string."

        value.GetString()
