module RNNTests

open Xunit
open FsUnit.Xunit
open Toro
open Toro.NN
open TestHelper

[<Fact>]
let ``LSTM zeroState produces correct shapes`` () =
    let lstm = LSTM.initDefault 10 20 F32 Cpu |> unwrap
    let state = lstm.zeroState 3 |> unwrap
    state.H.Shape |> should equal [ 3; 20 ]
    state.C.Shape |> should equal [ 3; 20 ]

[<Fact>]
let ``LSTM step produces correct output shape`` () =
    let lstm = LSTM.initDefault 10 20 F32 Cpu |> unwrap
    let state = lstm.zeroState 2 |> unwrap
    let input = Tensor.randn ([ 2; 10 ], F32, Cpu) |> unwrap
    let newState = lstm.step input state |> unwrap
    newState.H.Shape |> should equal [ 2; 20 ]
    newState.C.Shape |> should equal [ 2; 20 ]

[<Fact>]
let ``RNN.scan with LSTM processes full sequence`` () =
    let lstm = LSTM.initDefault 8 16 F32 Cpu |> unwrap
    let input = Tensor.randn ([ 2; 5; 8 ], F32, Cpu) |> unwrap
    let states = RNN.scan lstm.zeroState lstm.step input |> unwrap
    states.Length |> should equal 5
    states[4].H.Shape |> should equal [ 2; 16 ]

[<Fact>]
let ``RNN.scan with LSTM states can be stacked to tensor`` () =
    let lstm = LSTM.initDefault 8 16 F32 Cpu |> unwrap
    let input = Tensor.randn ([ 2; 5; 8 ], F32, Cpu) |> unwrap
    let states = RNN.scan lstm.zeroState lstm.step input |> unwrap
    let output = Tensor.stack (states |> List.map _.H, 1) |> unwrap
    output.Shape |> should equal [ 2; 5; 16 ]

[<Fact>]
let ``GRU zeroState produces correct shape`` () =
    let gru = GRU.initDefault 10 20 F32 Cpu |> unwrap
    let state = gru.zeroState 3 |> unwrap
    state.H.Shape |> should equal [ 3; 20 ]

[<Fact>]
let ``GRU step produces correct output shape`` () =
    let gru = GRU.initDefault 10 20 F32 Cpu |> unwrap
    let state = gru.zeroState 2 |> unwrap
    let input = Tensor.randn ([ 2; 10 ], F32, Cpu) |> unwrap
    let newState = gru.step input state |> unwrap
    newState.H.Shape |> should equal [ 2; 20 ]

[<Fact>]
let ``RNN.scan with GRU processes full sequence`` () =
    let gru = GRU.initDefault 8 16 F32 Cpu |> unwrap
    let input = Tensor.randn ([ 2; 5; 8 ], F32, Cpu) |> unwrap
    let states = RNN.scan gru.zeroState gru.step input |> unwrap
    states.Length |> should equal 5
    states[4].H.Shape |> should equal [ 2; 16 ]

[<Fact>]
let ``RNN.scan with GRU states can be stacked to tensor`` () =
    let gru = GRU.initDefault 8 16 F32 Cpu |> unwrap
    let input = Tensor.randn ([ 2; 5; 8 ], F32, Cpu) |> unwrap
    let states = RNN.scan gru.zeroState gru.step input |> unwrap
    let output = Tensor.stack (states |> List.map _.H, 1) |> unwrap
    output.Shape |> should equal [ 2; 5; 16 ]
