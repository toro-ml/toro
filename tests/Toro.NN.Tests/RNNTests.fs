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
let ``LSTM seq processes full sequence`` () =
    let lstm = LSTM.initDefault 8 16 F32 Cpu |> unwrap
    let input = Tensor.randn ([ 2; 5; 8 ], F32, Cpu) |> unwrap
    let states = lstm.seq input |> unwrap
    states.Length |> should equal 5
    states[4].H.Shape |> should equal [ 2; 16 ]

[<Fact>]
let ``LSTM statesToTensor stacks hidden states`` () =
    let lstm = LSTM.initDefault 8 16 F32 Cpu |> unwrap
    let input = Tensor.randn ([ 2; 5; 8 ], F32, Cpu) |> unwrap
    let states = lstm.seq input |> unwrap
    let output = lstm.statesToTensor states |> unwrap
    output.Shape |> should equal [ 2; 5; 16 ]

[<Fact>]
let ``LSTM implements IRNN interface`` () =
    let lstm = LSTM.initDefault 4 8 F32 Cpu |> unwrap
    let rnn = lstm :> IRNN<LSTMState>
    let state = rnn.zeroState 1 |> unwrap
    state.H.Shape |> should equal [ 1; 8 ]

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
let ``GRU seq processes full sequence`` () =
    let gru = GRU.initDefault 8 16 F32 Cpu |> unwrap
    let input = Tensor.randn ([ 2; 5; 8 ], F32, Cpu) |> unwrap
    let states = gru.seq input |> unwrap
    states.Length |> should equal 5
    states[4].H.Shape |> should equal [ 2; 16 ]

[<Fact>]
let ``GRU statesToTensor stacks hidden states`` () =
    let gru = GRU.initDefault 8 16 F32 Cpu |> unwrap
    let input = Tensor.randn ([ 2; 5; 8 ], F32, Cpu) |> unwrap
    let states = gru.seq input |> unwrap
    let output = gru.statesToTensor states |> unwrap
    output.Shape |> should equal [ 2; 5; 16 ]

[<Fact>]
let ``GRU implements IRNN interface`` () =
    let gru = GRU.initDefault 4 8 F32 Cpu |> unwrap
    let rnn = gru :> IRNN<GRUState>
    let state = rnn.zeroState 1 |> unwrap
    state.H.Shape |> should equal [ 1; 8 ]
