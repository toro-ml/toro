module TestHelper

open Toro

let unwrap r =
    match r with
    | Ok v -> v
    | Error e -> failwithf "Unexpected error: %A" e

let scalarF32 (t: Tensor) =
    (t.sumAll () |> unwrap).toFloat32Scalar () |> unwrap

let scalarF64 (t: Tensor) =
    (t.sumAll () |> unwrap).toFloat64Scalar () |> unwrap
