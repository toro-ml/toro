module PoolingTests

open Xunit
open FsUnit.Xunit
open Toro
open TorchSharp
open Toro.NN
open TestHelper

[<Fact>]
let ``MaxPool1d reduces temporal dimension`` () =
    let x = torch.randn ([| 1L; 3L; 20L |], dtype = torch.float32, device = torch.CPU)
    let pool = MaxPool1d.createDefault 2
    let y = pool.forward x
    y.shape |> should equal [| 1L; 3L; 10L |]

[<Fact>]
let ``MaxPool2d halves spatial dimensions`` () =
    let x =
        torch.randn ([| 1L; 1L; 8L; 8L |], dtype = torch.float32, device = torch.CPU)

    let pool = MaxPool2d.createDefault 2
    let y = pool.forward x
    y.shape |> should equal [| 1L; 1L; 4L; 4L |]

[<Fact>]
let ``MaxPool2d with stride and padding`` () =
    let x =
        torch.randn ([| 2L; 3L; 6L; 6L |], dtype = torch.float32, device = torch.CPU)

    let pool = MaxPool2d.create 3 1 1
    let y = pool.forward x
    y.shape |> should equal [| 2L; 3L; 6L; 6L |]

[<Fact>]
let ``AvgPool2d halves spatial dimensions`` () =
    let x =
        torch.randn ([| 1L; 1L; 8L; 8L |], dtype = torch.float32, device = torch.CPU)

    let pool = AvgPool2d.createDefault 2
    let y = pool.forward x
    y.shape |> should equal [| 1L; 1L; 4L; 4L |]

[<Fact>]
let ``MaxPool2d implements IModule`` () =
    let pool = MaxPool2d.createDefault 2
    let m = pool :> IModule

    let x =
        torch.randn ([| 1L; 1L; 4L; 4L |], dtype = torch.float32, device = torch.CPU)

    let y = m.forward x
    y.shape |> should equal [| 1L; 1L; 2L; 2L |]

[<Fact>]
let ``SequencePool.maskedMean averages unmasked tokens`` () =
    let hidden =
        torch.tensor (
            array2D [| [| 1.0f; 10.0f |]; [| 3.0f; 20.0f |]; [| 100.0f; 200.0f |] |],
            dtype = torch.float32,
            device = torch.CPU
        )
        |> fun t -> t.unsqueeze 0L

    let mask =
        torch.tensor ([| 1.0f; 1.0f; 0.0f |], dtype = torch.float32, device = torch.CPU)
        |> fun t -> t.unsqueeze 0L

    let pooled = SequencePool.maskedMean hidden mask
    pooled.shape |> should equal [| 1L; 2L |]

    let values = pooled.data<float32>().ToArray()
    values[0] |> should (equalWithin 1e-5f) 2.0f
    values[1] |> should (equalWithin 1e-5f) 15.0f
