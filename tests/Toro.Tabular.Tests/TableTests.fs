module TableTests

open System
open Xunit
open FsUnit.Xunit
open TorchSharp
open Toro
open Toro.Tabular

let private floats (values: float32 array) =
    torch.tensor (values, dtype = torch.float32, device = torch.CPU)

let private ints (values: int64 array) =
    torch.tensor (values, dtype = torch.int64, device = torch.CPU)

[<Fact>]
let ``Table.create keeps a shared row count`` () =
    let table =
        Table.create [ "tf", Floats(floats [| 1.0f; 0.5f |]); "len", Ints(ints [| 3L; 5L |]) ]

    table.Length |> should equal 2L
    table.Columns.Count |> should equal 2

[<Fact>]
let ``Table.create rejects a length mismatch`` () =
    Assert.Throws<ArgumentException>(fun () ->
        Table.create [ "tf", Floats(floats [| 1.0f; 0.5f |]); "len", Ints(ints [| 3L |]) ]
        |> ignore)
    |> ignore

[<Fact>]
let ``Table.create rejects an empty column list`` () =
    Assert.Throws<ArgumentException>(fun () -> Table.create [] |> ignore)
    |> ignore

[<Fact>]
let ``Table.create rejects duplicate names`` () =
    Assert.Throws<ArgumentException>(fun () ->
        Table.create [ "tf", Floats(floats [| 1.0f |]); "tf", Ints(ints [| 1L |]) ]
        |> ignore)
    |> ignore

[<Fact>]
let ``Table.column returns a named column`` () =
    let tensor = floats [| 2.0f |]
    let table = Table.create [ "tf", Floats tensor ]

    match Table.column "tf" table with
    | Floats value -> value.sum().ToSingle() |> should (equalWithin 1e-5f) 2.0f
    | _ -> failwith "expected Floats"

[<Fact>]
let ``Table.column rejects a missing name`` () =
    let table = Table.create [ "tf", Floats(floats [| 1.0f |]) ]

    Assert.Throws<ArgumentException>(fun () -> Table.column "missing" table |> ignore)
    |> ignore

[<Fact>]
let ``Table.features stacks 1-d numeric columns`` () =
    let table =
        Table.create [ "tf", Floats(floats [| 1.0f; 0.0f |]); "len", Ints(ints [| 3L; 5L |]) ]

    let stacked = Table.features [ "tf"; "len" ] table
    stacked.shape |> should equal [| 2L; 2L |]
    stacked.dtype |> should equal torch.float32

    let values = stacked.data<float32>().ToArray()
    values[0] |> should (equalWithin 1e-5f) 1.0f
    values[1] |> should (equalWithin 1e-5f) 3.0f
    values[2] |> should (equalWithin 1e-5f) 0.0f
    values[3] |> should (equalWithin 1e-5f) 5.0f

[<Fact>]
let ``Table.features rejects a vector column`` () =
    let vectors = torch.zeros ([| 2L; 3L |], dtype = torch.float32, device = torch.CPU)

    let table = Table.create [ "emb", Vectors vectors ]

    Assert.Throws<ArgumentException>(fun () -> Table.features [ "emb" ] table |> ignore)
    |> ignore

[<Fact>]
let ``Table.features rejects a missing name`` () =
    let table = Table.create [ "tf", Floats(floats [| 1.0f |]) ]

    Assert.Throws<ArgumentException>(fun () -> Table.features [ "len" ] table |> ignore)
    |> ignore
