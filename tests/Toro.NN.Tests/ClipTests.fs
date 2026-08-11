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

        |> fun t -> t.requiresGrad ()

    let loss = w.mul w |> (fun t -> t.sumAll ())
    loss.backward ()

    let totalNorm = Clip.gradNorm 0.5 [ w ]
    totalNorm |> should be (greaterThan 0.0)

    let g = w.grad ()
    let normSq = g.mul g |> (fun t -> t.sumAll ())
    let normAfter = (normSq.toFloat64Scalar ()) |> sqrt

    normAfter |> should be (lessThan 0.51)

[<Fact>]
let ``clipGradNorm returns norm without clipping when below max`` () =
    let w =
        Tensor.ofList ([ 0.1f; 0.1f ], Cpu)

        |> fun t -> t.requiresGrad ()

    let loss = w.mul w |> (fun t -> t.sumAll ())
    loss.backward ()

    let totalNorm = Clip.gradNorm 100.0 [ w ]
    totalNorm |> should be (lessThan 1.0)

[<Fact>]
let ``clipGradValue clamps gradient elements`` () =
    let w =
        Tensor.randn ([ 10; 10 ], F32, Cpu)

        |> fun t -> t.requiresGrad ()

    let loss =
        w.mul w

        |> fun t -> t.mul w |> (fun t -> t.sumAll ())

    loss.backward ()

    Clip.gradValue 0.1 [ w ]

    let g = w.grad ()
    let clampCheck = g.clamp (-0.1, 0.1)
    let diff = (g - clampCheck).abs ()
    let maxDiff = (diff.sumAll ()).toFloat32Scalar ()
    maxDiff |> should be (lessThan 1e-6f)
