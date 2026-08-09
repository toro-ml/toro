module ClipTests

open Xunit
open FsUnit.Xunit
open Toro
open Toro.NN
open TestHelper

[<Fact>]
let ``clipGradNorm scales gradients when norm exceeds max`` () =
    let w =
        Tensor.randn ([ 4; 4 ], F32, Cpu)
        |> unwrap
        |> fun t -> t.requiresGrad () |> unwrap

    let loss = w.mul w |> unwrap |> (fun t -> t.sumAll () |> unwrap)
    loss.backward () |> unwrap

    let totalNorm = Clip.gradNorm 0.5 [ w ] |> unwrap
    totalNorm |> should be (greaterThan 0.0)

    let g = w.grad () |> unwrap
    let normSq = g.mul g |> unwrap |> (fun t -> t.sumAll () |> unwrap)
    let normAfter = (normSq.toFloat64Scalar () |> unwrap) |> sqrt

    normAfter |> should be (lessThan 0.51)

[<Fact>]
let ``clipGradNorm returns norm without clipping when below max`` () =
    let w =
        Tensor.ofList ([ 0.1f; 0.1f ], Cpu)
        |> unwrap
        |> fun t -> t.requiresGrad () |> unwrap

    let loss = w.mul w |> unwrap |> (fun t -> t.sumAll () |> unwrap)
    loss.backward () |> unwrap

    let totalNorm = Clip.gradNorm 100.0 [ w ] |> unwrap
    totalNorm |> should be (lessThan 1.0)

[<Fact>]
let ``clipGradValue clamps gradient elements`` () =
    let w =
        Tensor.randn ([ 10; 10 ], F32, Cpu)
        |> unwrap
        |> fun t -> t.requiresGrad () |> unwrap

    let loss =
        w.mul w
        |> unwrap
        |> fun t -> t.mul w |> unwrap |> (fun t -> t.sumAll () |> unwrap)

    loss.backward () |> unwrap

    Clip.gradValue 0.1 [ w ] |> unwrap

    let g = w.grad () |> unwrap
    let clampCheck = g.clamp (-0.1, 0.1) |> unwrap
    let diff = (g - clampCheck).abs () |> unwrap
    let maxDiff = (diff.sumAll () |> unwrap).toFloat32Scalar () |> unwrap
    maxDiff |> should be (lessThan 1e-6f)
