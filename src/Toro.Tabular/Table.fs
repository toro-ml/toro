namespace Toro.Tabular

open TorchSharp
open Toro

/// A named table column. `Floats` and `Ints` are length `n`; `Vectors` is `[n, d]`.
type Column =
    | Floats of Tensor
    | Ints of Tensor
    | Vectors of Tensor

/// A table of named columns with a shared row count.
type Table = {
    Length: int64
    Columns: Map<string, Column>
}

module private ColumnOps =
    let length (name: string) (column: Column) : int64 =
        match column with
        | Floats tensor
        | Ints tensor ->
            if tensor.dim () <> 1 then
                invalidArg (nameof column) $"Column '{name}' must be 1-d, but has {tensor.dim ()} dimensions."

            tensor.shape[0]
        | Vectors tensor ->
            if tensor.dim () <> 2 then
                invalidArg (nameof column) $"Column '{name}' must be 2-d, but has {tensor.dim ()} dimensions."

            tensor.shape[0]

    let asFeature (name: string) (column: Column) : Tensor =
        match column with
        | Floats tensor
        | Ints tensor -> tensor.to_type(torch.float32).unsqueeze 1L
        | Vectors _ -> invalidArg (nameof name) $"Column '{name}' is a vector column; features requires 1-d numeric columns."

/// Constructors and queries for named-column tables.
module Table =

    /// Create a table from named columns. All columns must share the same row count.
    let create (columns: (string * Column) list) : Table =
        if columns.IsEmpty then
            invalidArg (nameof columns) "A table must contain at least one column."

        let duplicates =
            columns
            |> List.map fst
            |> List.groupBy id
            |> List.choose (fun (name, copies) -> if copies.Length > 1 then Some name else None)

        match duplicates with
        | name :: _ -> invalidArg (nameof columns) $"Duplicate column name '{name}'."
        | [] -> ()

        let lengths =
            columns
            |> List.map (fun (name, column) -> name, ColumnOps.length name column)

        let n = snd lengths.Head

        lengths
        |> List.iter (fun (name, length) ->
            if length <> n then
                invalidArg (nameof columns) $"Column '{name}' has length {length}, expected {n}.")

        {
            Length = n
            Columns = Map.ofList columns
        }

    /// Return the column named `name`.
    let column (name: string) (table: Table) : Column =
        match Map.tryFind name table.Columns with
        | Some value -> value
        | None -> invalidArg (nameof name) $"Column '{name}' was not found."

    /// Stack the named 1-d numeric columns into a `[n, f]` float32 tensor.
    let features (names: string list) (table: Table) : Tensor =
        if names.IsEmpty then
            invalidArg (nameof names) "At least one feature column is required."

        names
        |> List.map (fun name -> ColumnOps.asFeature name (column name table))
        |> fun stacked -> torch.cat (Array.ofList stacked, 1L)
