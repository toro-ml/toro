module VisionTests

open Xunit
open FsUnit.Xunit
open Toro
open TorchSharp
open Toro.Vision
open TestHelper

[<Fact>]
let ``Normalize produces correct channel statistics`` () =
    let x = torch.ones ([| 3L; 4L; 4L |], dtype = torch.float32, device = torch.CPU)

    let norm: ITransform = {
        Normalize.Mean = [ 0.5; 0.5; 0.5 ]
        Std = [ 0.5; 0.5; 0.5 ]
    }

    let out = norm.apply x
    out.shape |> should equal [| 3L; 4L; 4L |]

    out.at [ I 0; I 0; I 0 ]
    |> scalarF32
    |> should (equalWithin 1e-5) 1.0f

[<Fact>]
let ``Normalize batched produces correct shape`` () =
    let x = torch.ones ([| 2L; 3L; 4L; 4L |], dtype = torch.float32, device = torch.CPU)

    let norm: ITransform = {
        Normalize.Mean = [ 0.5; 0.5; 0.5 ]
        Std = [ 0.5; 0.5; 0.5 ]
    }

    let out = norm.apply x
    out.shape |> should equal [| 2L; 3L; 4L; 4L |]

[<Fact>]
let ``Resize produces correct shape`` () =
    let x = torch.randn ([| 3L; 32L; 32L |], dtype = torch.float32, device = torch.CPU)
    let resize = Resize.create 16 16
    let out = resize.apply x
    out.shape |> should equal [| 3L; 16L; 16L |]

[<Fact>]
let ``Resize batched produces correct shape`` () =
    let x =
        torch.randn ([| 2L; 3L; 32L; 32L |], dtype = torch.float32, device = torch.CPU)

    let resize = Resize.create 16 16
    let out = resize.apply x
    out.shape |> should equal [| 2L; 3L; 16L; 16L |]

[<Fact>]
let ``RandomCrop produces correct shape`` () =
    let x = torch.randn ([| 3L; 32L; 32L |], dtype = torch.float32, device = torch.CPU)
    let crop = RandomCrop.create 16 16
    let out = crop.apply x
    out.shape |> should equal [| 3L; 16L; 16L |]

[<Fact>]
let ``RandomCrop rejects too-small input`` () =
    let x = torch.randn ([| 3L; 8L; 8L |], dtype = torch.float32, device = torch.CPU)
    let crop = RandomCrop.create 16 16

    try
        crop.apply x |> ignore
        failwith "Expected exception for too-small input"
    with _ ->
        ()

[<Fact>]
let ``RandomHorizontalFlip preserves shape`` () =
    let x = torch.randn ([| 3L; 32L; 32L |], dtype = torch.float32, device = torch.CPU)
    let flip = RandomHorizontalFlip.defaultFlip
    let out = flip.apply x
    out.shape |> should equal [| 3L; 32L; 32L |]

[<Fact>]
let ``Compose chains transforms`` () =
    let x = torch.randn ([| 3L; 32L; 32L |], dtype = torch.float32, device = torch.CPU)

    let transforms: ITransform list = [
        Resize.create 16 16 :> ITransform
        {
            Normalize.Mean = [ 0.5; 0.5; 0.5 ]
            Std = [ 0.5; 0.5; 0.5 ]
        }
    ]

    let out = Compose.apply transforms x
    out.shape |> should equal [| 3L; 16L; 16L |]
