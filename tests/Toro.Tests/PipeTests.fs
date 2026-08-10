module PipeTests

open Xunit
open Toro
open TestHelper

let double x : Result<int, ToroError> = Ok(x * 2)
let addOne x : Result<int, ToroError> = Ok(x + 1)
let toString x : Result<string, ToroError> = Ok(string x)

let failIfNeg x : Result<int, ToroError> =
    if x < 0 then Error(Msg "negative") else Ok x

[<Fact>]
let ``chain two functions`` () =
    let f = double >=> addOne
    Assert.Equal(7, f 3 |> unwrap)

[<Fact>]
let ``chain three functions`` () =
    let f = double >=> addOne >=> double
    Assert.Equal(14, f 3 |> unwrap)

[<Fact>]
let ``short-circuits on error`` () =
    let f = failIfNeg >=> double >=> addOne

    match f -1 with
    | Error(Msg "negative") -> ()
    | other -> Assert.Fail $"Expected Error(Msg \"negative\"), got %A{other}"

[<Fact>]
let ``error in middle stops chain`` () =
    let f = double >=> (fun x -> failIfNeg (x - 100)) >=> addOne

    match f 3 with
    | Error(Msg "negative") -> ()
    | other -> Assert.Fail $"Expected Error(Msg \"negative\"), got %A{other}"

[<Fact>]
let ``composes across different types`` () =
    let f = double >=> toString
    Assert.Equal("10", f 5 |> unwrap)
