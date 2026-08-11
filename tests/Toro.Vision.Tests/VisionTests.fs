module VisionTests

open Xunit
open FsUnit.Xunit
open Toro
open Toro.Vision
open TestHelper

[<Fact>]
let ``Normalize produces correct channel statistics`` () =
    let x = Tensor.ones ([ 3; 4; 4 ], F32, Cpu)

    let norm: ITransform = {
        Normalize.Mean = [ 0.5; 0.5; 0.5 ]
        Std = [ 0.5; 0.5; 0.5 ]
    }

    let out = norm.apply x
    out.Shape |> should equal [ 3; 4; 4 ]

    out.at [ I 0; I 0; I 0 ]
    |> scalarF32
    |> should (equalWithin 1e-5) 1.0f

[<Fact>]
let ``Normalize batched produces correct shape`` () =
    let x = Tensor.ones ([ 2; 3; 4; 4 ], F32, Cpu)

    let norm: ITransform = {
        Normalize.Mean = [ 0.5; 0.5; 0.5 ]
        Std = [ 0.5; 0.5; 0.5 ]
    }

    let out = norm.apply x
    out.Shape |> should equal [ 2; 3; 4; 4 ]

[<Fact>]
let ``Resize produces correct shape`` () =
    let x = Tensor.randn ([ 3; 32; 32 ], F32, Cpu)
    let resize = Resize.create 16 16
    let out = resize.apply x
    out.Shape |> should equal [ 3; 16; 16 ]

[<Fact>]
let ``Resize batched produces correct shape`` () =
    let x = Tensor.randn ([ 2; 3; 32; 32 ], F32, Cpu)
    let resize = Resize.create 16 16
    let out = resize.apply x
    out.Shape |> should equal [ 2; 3; 16; 16 ]

[<Fact>]
let ``RandomCrop produces correct shape`` () =
    let x = Tensor.randn ([ 3; 32; 32 ], F32, Cpu)
    let crop = RandomCrop.create 16 16
    let out = crop.apply x
    out.Shape |> should equal [ 3; 16; 16 ]

[<Fact>]
let ``RandomCrop rejects too-small input`` () =
    let x = Tensor.randn ([ 3; 8; 8 ], F32, Cpu)
    let crop = RandomCrop.create 16 16

    try
        crop.apply x |> ignore
        failwith "Expected exception for too-small input"
    with _ ->
        ()

[<Fact>]
let ``RandomHorizontalFlip preserves shape`` () =
    let x = Tensor.randn ([ 3; 32; 32 ], F32, Cpu)
    let flip = RandomHorizontalFlip.defaultFlip
    let out = flip.apply x
    out.Shape |> should equal [ 3; 32; 32 ]

[<Fact>]
let ``Compose chains transforms`` () =
    let x = Tensor.randn ([ 3; 32; 32 ], F32, Cpu)

    let transforms: ITransform list = [
        Resize.create 16 16 :> ITransform
        {
            Normalize.Mean = [ 0.5; 0.5; 0.5 ]
            Std = [ 0.5; 0.5; 0.5 ]
        }
    ]

    let out = Compose.apply transforms x
    out.Shape |> should equal [ 3; 16; 16 ]
