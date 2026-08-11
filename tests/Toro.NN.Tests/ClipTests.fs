module ClipTests

open Xunit
open FsUnit.Xunit
open Toro
open TorchSharp
open Toro.NN
open TestHelper

[<Fact>]
let ``clipGradNorm scales gradients when norm exceeds max`` () =
    let w =
        torch.randn ([| 4L; 4L |], dtype = torch.float32, device = torch.CPU)

        |> fun t -> t.requires_grad_ ()

    let loss = w.mul w |> (fun t -> t.sum ())
    loss.backward ()

    let totalNorm = Clip.gradNorm 0.5 [ w ]
    totalNorm |> should be (greaterThan 0.0)

    let g = w.grad ()
    let normSq = g.mul g |> (fun t -> t.sum ())
    let normAfter = (normSq.ToDouble()) |> sqrt

    normAfter |> should be (lessThan 0.51)

[<Fact>]
let ``clipGradNorm returns norm without clipping when below max`` () =
    let w =
        torch.tensor ([| 0.1f; 0.1f |], device = torch.CPU)

        |> fun t -> t.requires_grad_ ()

    let loss = w.mul w |> (fun t -> t.sum ())
    loss.backward ()

    let totalNorm = Clip.gradNorm 100.0 [ w ]
    totalNorm |> should be (lessThan 1.0)

[<Fact>]
let ``clipGradValue clamps gradient elements`` () =
    let w =
        torch.randn ([| 10L; 10L |], dtype = torch.float32, device = torch.CPU)

        |> fun t -> t.requires_grad_ ()

    let loss =
        w.mul w

        |> fun t -> t.mul w |> (fun t -> t.sum ())

    loss.backward ()

    Clip.gradValue 0.1 [ w ]

    let g = w.grad ()
    let clampCheck = g.clamp (scalar (-0.1), scalar 0.1)
    let diff = (g - clampCheck).abs ()
    let maxDiff = (diff.sum ()).ToSingle()
    maxDiff |> should be (lessThan 1e-6f)
