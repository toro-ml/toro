module MetricsTests

open Xunit
open FsUnit.Xunit
open Toro
open Toro.NN
open TestHelper

[<Fact>]
let ``accuracy of identical predictions is 1`` () =
    let pred = Tensor.ofList ([ 0L; 1L; 2L; 0L ], Cpu) |> unwrap
    let target = Tensor.ofList ([ 0L; 1L; 2L; 0L ], Cpu) |> unwrap

    let acc = Metrics.accuracy pred target |> unwrap
    acc |> should (equalWithin 1e-6) 1.0

[<Fact>]
let ``accuracy of all wrong predictions is 0`` () =
    let pred = Tensor.ofList ([ 0L; 0L; 0L; 0L ], Cpu) |> unwrap
    let target = Tensor.ofList ([ 1L; 1L; 1L; 1L ], Cpu) |> unwrap

    let acc = Metrics.accuracy pred target |> unwrap
    acc |> should (equalWithin 1e-6) 0.0

[<Fact>]
let ``accuracy of half correct is 0.5`` () =
    let pred = Tensor.ofList ([ 0L; 1L; 0L; 1L ], Cpu) |> unwrap
    let target = Tensor.ofList ([ 0L; 1L; 1L; 0L ], Cpu) |> unwrap

    let acc = Metrics.accuracy pred target |> unwrap
    acc |> should (equalWithin 1e-6) 0.5

[<Fact>]
let ``accuracyFromLogits takes argmax`` () =
    let logits =
        Tensor.ofList ([ [ 2.0f; 0.1f; 0.1f ]; [ 0.1f; 2.0f; 0.1f ] ], Cpu)
        |> unwrap

    let target = Tensor.ofList ([ 0L; 1L ], Cpu) |> unwrap

    let acc = Metrics.accuracyFromLogits logits target |> unwrap
    acc |> should (equalWithin 1e-6) 1.0

[<Fact>]
let ``precision for perfect predictions`` () =
    let pred = Tensor.ofList ([ 0L; 1L; 1L; 0L ], Cpu) |> unwrap
    let target = Tensor.ofList ([ 0L; 1L; 1L; 0L ], Cpu) |> unwrap

    let p = Metrics.precision 2 pred target |> unwrap
    p.[0] |> should (equalWithin 1e-6) 1.0
    p.[1] |> should (equalWithin 1e-6) 1.0

[<Fact>]
let ``recall for perfect predictions`` () =
    let pred = Tensor.ofList ([ 0L; 1L; 1L; 0L ], Cpu) |> unwrap
    let target = Tensor.ofList ([ 0L; 1L; 1L; 0L ], Cpu) |> unwrap

    let r = Metrics.recall 2 pred target |> unwrap
    r.[0] |> should (equalWithin 1e-6) 1.0
    r.[1] |> should (equalWithin 1e-6) 1.0

[<Fact>]
let ``f1 for perfect predictions`` () =
    let pred = Tensor.ofList ([ 0L; 1L; 1L; 0L ], Cpu) |> unwrap
    let target = Tensor.ofList ([ 0L; 1L; 1L; 0L ], Cpu) |> unwrap

    let f = Metrics.f1 2 pred target |> unwrap
    f.[0] |> should (equalWithin 1e-6) 1.0
    f.[1] |> should (equalWithin 1e-6) 1.0

[<Fact>]
let ``precision with false positives`` () =
    let pred = Tensor.ofList ([ 1L; 1L; 1L; 1L ], Cpu) |> unwrap
    let target = Tensor.ofList ([ 0L; 0L; 1L; 1L ], Cpu) |> unwrap

    let p = Metrics.precision 2 pred target |> unwrap
    p.[1] |> should (equalWithin 1e-6) 0.5

[<Fact>]
let ``recall with false negatives`` () =
    let pred = Tensor.ofList ([ 0L; 0L; 0L; 0L ], Cpu) |> unwrap
    let target = Tensor.ofList ([ 0L; 0L; 1L; 1L ], Cpu) |> unwrap

    let r = Metrics.recall 2 pred target |> unwrap
    r.[0] |> should (equalWithin 1e-6) 1.0
    r.[1] |> should (equalWithin 1e-6) 0.0
