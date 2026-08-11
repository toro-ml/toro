module RNNTests

open Xunit
open FsUnit.Xunit
open Toro
open TorchSharp
open Toro.NN
open TestHelper

[<Fact>]
let ``LSTM zeroState produces correct shapes`` () =
    let lstm = LSTM.initDefault 10 20 torch.float32 torch.CPU
    let state = lstm.zeroState 3
    state.H.shape |> should equal [| 3L; 20L |]
    state.C.shape |> should equal [| 3L; 20L |]

[<Fact>]
let ``LSTM step produces correct output shape`` () =
    let lstm = LSTM.initDefault 10 20 torch.float32 torch.CPU
    let state = lstm.zeroState 2
    let input = torch.randn ([| 2L; 10L |], dtype = torch.float32, device = torch.CPU)
    let newState = lstm.step input state
    newState.H.shape |> should equal [| 2L; 20L |]
    newState.C.shape |> should equal [| 2L; 20L |]

[<Fact>]
let ``RNN.scan with LSTM processes full sequence`` () =
    let lstm = LSTM.initDefault 8 16 torch.float32 torch.CPU

    let input =
        torch.randn ([| 2L; 5L; 8L |], dtype = torch.float32, device = torch.CPU)

    let states = RNN.scan lstm.zeroState lstm.step input
    states.Length |> should equal 5
    states[4].H.shape |> should equal [| 2L; 16L |]

[<Fact>]
let ``RNN.scan with LSTM states can be stacked to tensor`` () =
    let lstm = LSTM.initDefault 8 16 torch.float32 torch.CPU

    let input =
        torch.randn ([| 2L; 5L; 8L |], dtype = torch.float32, device = torch.CPU)

    let states = RNN.scan lstm.zeroState lstm.step input
    let output = torch.stack (states |> List.map _.H |> List.toArray, dim = 1L)
    output.shape |> should equal [| 2L; 5L; 16L |]

[<Fact>]
let ``GRU zeroState produces correct shape`` () =
    let gru = GRU.initDefault 10 20 torch.float32 torch.CPU
    let state = gru.zeroState 3
    state.H.shape |> should equal [| 3L; 20L |]

[<Fact>]
let ``GRU step produces correct output shape`` () =
    let gru = GRU.initDefault 10 20 torch.float32 torch.CPU
    let state = gru.zeroState 2
    let input = torch.randn ([| 2L; 10L |], dtype = torch.float32, device = torch.CPU)
    let newState = gru.step input state
    newState.H.shape |> should equal [| 2L; 20L |]

[<Fact>]
let ``RNN.scan with GRU processes full sequence`` () =
    let gru = GRU.initDefault 8 16 torch.float32 torch.CPU

    let input =
        torch.randn ([| 2L; 5L; 8L |], dtype = torch.float32, device = torch.CPU)

    let states = RNN.scan gru.zeroState gru.step input
    states.Length |> should equal 5
    states[4].H.shape |> should equal [| 2L; 16L |]

[<Fact>]
let ``RNN.scan with GRU states can be stacked to tensor`` () =
    let gru = GRU.initDefault 8 16 torch.float32 torch.CPU

    let input =
        torch.randn ([| 2L; 5L; 8L |], dtype = torch.float32, device = torch.CPU)

    let states = RNN.scan gru.zeroState gru.step input
    let output = torch.stack (states |> List.map _.H |> List.toArray, dim = 1L)
    output.shape |> should equal [| 2L; 5L; 16L |]
