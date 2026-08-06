module RNNTests

open Xunit
open FsUnit.Xunit
open Toro
open Toro.NN
open TestHelper

[<Fact>]
let ``LSTM zeroState produces correct shapes`` () =
    let vb = VarBuilder.fromInit F32 Cpu
    let lstm = LSTM.createDefault 10 20 vb |> unwrap
    let state = lstm.zeroState 3 |> unwrap
    state.H.Shape |> should equal [ 3; 20 ]
    state.C.Shape |> should equal [ 3; 20 ]

[<Fact>]
let ``LSTM step produces correct output shape`` () =
    let vb = VarBuilder.fromInit F32 Cpu
    let lstm = LSTM.createDefault 10 20 vb |> unwrap
    let state = lstm.zeroState 2 |> unwrap
    let input = Tensor.randn ([ 2; 10 ], F32, Cpu) |> unwrap
    let newState = lstm.step input state |> unwrap
    newState.H.Shape |> should equal [ 2; 20 ]
    newState.C.Shape |> should equal [ 2; 20 ]

[<Fact>]
let ``LSTM seq processes full sequence`` () =
    let vb = VarBuilder.fromInit F32 Cpu
    let lstm = LSTM.createDefault 8 16 vb |> unwrap
    let input = Tensor.randn ([ 2; 5; 8 ], F32, Cpu) |> unwrap
    let states = lstm.seq input |> unwrap
    states.Length |> should equal 5
    states[4].H.Shape |> should equal [ 2; 16 ]

[<Fact>]
let ``LSTM statesToTensor stacks hidden states`` () =
    let vb = VarBuilder.fromInit F32 Cpu
    let lstm = LSTM.createDefault 8 16 vb |> unwrap
    let input = Tensor.randn ([ 2; 5; 8 ], F32, Cpu) |> unwrap
    let states = lstm.seq input |> unwrap
    let output = lstm.statesToTensor states |> unwrap
    output.Shape |> should equal [ 2; 5; 16 ]

[<Fact>]
let ``LSTM implements IRNN interface`` () =
    let vb = VarBuilder.fromInit F32 Cpu
    let lstm = LSTM.createDefault 4 8 vb |> unwrap
    let rnn = lstm :> IRNN<LSTMState>
    let state = rnn.zeroState 1 |> unwrap
    state.H.Shape |> should equal [ 1; 8 ]

[<Fact>]
let ``GRU zeroState produces correct shape`` () =
    let vb = VarBuilder.fromInit F32 Cpu
    let gru = GRU.createDefault 10 20 vb |> unwrap
    let state = gru.zeroState 3 |> unwrap
    state.H.Shape |> should equal [ 3; 20 ]

[<Fact>]
let ``GRU step produces correct output shape`` () =
    let vb = VarBuilder.fromInit F32 Cpu
    let gru = GRU.createDefault 10 20 vb |> unwrap
    let state = gru.zeroState 2 |> unwrap
    let input = Tensor.randn ([ 2; 10 ], F32, Cpu) |> unwrap
    let newState = gru.step input state |> unwrap
    newState.H.Shape |> should equal [ 2; 20 ]

[<Fact>]
let ``GRU seq processes full sequence`` () =
    let vb = VarBuilder.fromInit F32 Cpu
    let gru = GRU.createDefault 8 16 vb |> unwrap
    let input = Tensor.randn ([ 2; 5; 8 ], F32, Cpu) |> unwrap
    let states = gru.seq input |> unwrap
    states.Length |> should equal 5
    states[4].H.Shape |> should equal [ 2; 16 ]

[<Fact>]
let ``GRU statesToTensor stacks hidden states`` () =
    let vb = VarBuilder.fromInit F32 Cpu
    let gru = GRU.createDefault 8 16 vb |> unwrap
    let input = Tensor.randn ([ 2; 5; 8 ], F32, Cpu) |> unwrap
    let states = gru.seq input |> unwrap
    let output = gru.statesToTensor states |> unwrap
    output.Shape |> should equal [ 2; 5; 16 ]

[<Fact>]
let ``GRU implements IRNN interface`` () =
    let vb = VarBuilder.fromInit F32 Cpu
    let gru = GRU.createDefault 4 8 vb |> unwrap
    let rnn = gru :> IRNN<GRUState>
    let state = rnn.zeroState 1 |> unwrap
    state.H.Shape |> should equal [ 1; 8 ]
