module AutogradTests

open Xunit
open FsUnit.Xunit
open Toro
open TorchSharp
open TestHelper

[<Fact>]
let ``backward computes gradients`` () =
    let x = torch.ones ([| 3L |], dtype = torch.float32, device = torch.CPU)
    let x = x.requires_grad_ ()

    let y = x.mul (scalar 3.0)
    let loss = y.sum ()

    loss.backward ()

    let g = x.grad ()
    scalarF32 g |> should equal 9.0f

[<Fact>]
let ``grad returns zero-like before backward`` () =
    let x = torch.ones ([| 2L |], dtype = torch.float32, device = torch.CPU)
    let x = x.requires_grad_ ()

    x.zeroGrad ()
    let g = x.grad ()
    scalarF32 g |> should equal 0.0f

[<Fact>]
let ``detach removes gradient tracking`` () =
    let x = torch.randn ([| 2L; 3L |], dtype = torch.float32, device = torch.CPU)
    let x = x.requires_grad_ ()
    let d = x.detach ()
    d.shape |> should equal [| 2L; 3L |]

[<Fact>]
let ``noGrad disables gradient tracking`` () =
    let t = torch.randn ([| 2L; 3L |], dtype = torch.float32, device = torch.CPU)
    let t = t.requires_grad_ ()

    let y =
        Toro.noGrad (fun () ->
            let r = t.mul t
            r)

    y.shape |> should equal [| 2L; 3L |]
