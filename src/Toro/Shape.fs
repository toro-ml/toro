namespace Toro

/// Utilities for tensor shapes represented as <c>int list</c>.
module Shape =
    let rank (shape: int list) = shape.Length

    let elemCount (shape: int list) =
        if shape.IsEmpty then
            0L
        else
            shape |> List.fold (fun acc d -> acc * int64 d) 1L

    let toInt64Array (shape: int list) =
        let arr = Array.zeroCreate shape.Length
        let mutable i = 0

        for d in shape do
            arr[i] <- int64 d
            i <- i + 1

        arr

    let ofInt64Array (shape: int64 array) =
        let mutable acc = []

        for i in shape.Length - 1 .. -1 .. 0 do
            acc <- int shape[i] :: acc

        acc
