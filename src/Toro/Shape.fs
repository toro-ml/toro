namespace Toro

/// Utilities for tensor shapes represented as <c>int list</c>.
module Shape =
    /// Return the number of dimensions.
    let rank (shape: int list) = shape.Length

    /// Return the total number of elements (product of dimensions).
    let elemCount (shape: int list) =
        if shape.IsEmpty then
            0L
        else
            shape |> List.fold (fun acc d -> acc * int64 d) 1L

    /// Convert to an int64 array.
    let toInt64Array (shape: int list) = shape |> List.map int64 |> List.toArray

    /// Convert from an int64 array.
    let ofInt64Array (shape: int64 array) = shape |> Array.map int |> Array.toList
