namespace Toro.NN

open Toro
open Toro.TensorOp

type IRNN<'State> =
    abstract zeroState: batchDim: int -> Result<'State, ToroError>
    abstract step: Tensor -> 'State -> Result<'State, ToroError>
    abstract seq: Tensor -> Result<'State list, ToroError>
    abstract statesToTensor: 'State list -> Result<Tensor, ToroError>

// --- LSTM ---

type LSTMState = { H: Tensor; C: Tensor }

type LSTMConfig = {
    WIhInit: Init
    WHhInit: Init
    BIhInit: Init option
    BHhInit: Init option
} with

    static member defaultConfig = {
        WIhInit = Init.KaimingNormal
        WHhInit = Init.KaimingNormal
        BIhInit = Some(Init.Const 0.0)
        BHhInit = Some(Init.Const 0.0)
    }

    static member defaultNoBias = {
        WIhInit = Init.KaimingNormal
        WHhInit = Init.KaimingNormal
        BIhInit = None
        BHhInit = None
    }

type LSTM = {
    WIh: Tensor
    WHh: Tensor
    BIh: Tensor option
    BHh: Tensor option
    HiddenDim: int
    DType: DType
    Device: Device
} with

    member this.zeroState(batchDim: int) : Result<LSTMState, ToroError> =
        result {
            let! zeros = Tensor.zeros ([ batchDim; this.HiddenDim ], this.DType, this.Device)

            let! zeros2 = zeros.clone ()
            return { H = zeros; C = zeros2 }
        }

    member this.step (input: Tensor) (state: LSTMState) : Result<LSTMState, ToroError> =
        result {
            let! wIhT = this.WIh.t ()
            let! wHhT = this.WHh.t ()
            let! ihGates = input.matmul wIhT
            let! hhGates = state.H.matmul wHhT

            let! ihGates =
                match this.BIh with
                | Some b -> ihGates.add b
                | None -> Ok ihGates

            let! hhGates =
                match this.BHh with
                | Some b -> hhGates.add b
                | None -> Ok hhGates

            let! gates = ihGates.add hhGates
            let! chunks = gates.chunk (4, 1)

            let! inGate = chunks[0].sigmoid ()
            let! forgetGate = chunks[1].sigmoid ()
            let! cellGate = chunks[2].tanh ()
            let! outGate = chunks[3].sigmoid ()

            let! nextC = forgetGate.mul state.C +~ inGate.mul cellGate
            let! nextH = outGate *~ nextC.tanh ()

            return { H = nextH; C = nextC }
        }

    member this.seq(input: Tensor) : Result<LSTMState list, ToroError> =
        result {
            let batchDim = input.Shape[0]
            let seqLen = input.Shape[1]
            let! initState = this.zeroState batchDim
            let mutable state = initState
            let output = System.Collections.Generic.List<LSTMState>()

            for i in 0 .. seqLen - 1 do
                let step_input = input.at [ A; I i ]
                let! newState = this.step step_input state
                state <- newState
                output.Add newState

            return output |> Seq.toList
        }

    member _.statesToTensor(states: LSTMState list) : Result<Tensor, ToroError> =
        let hs = states |> List.map _.H
        Tensor.stack (hs, 1)

    interface IRNN<LSTMState> with
        member this.zeroState batchDim = this.zeroState batchDim
        member this.step input state = this.step input state
        member this.seq input = this.seq input
        member this.statesToTensor states = this.statesToTensor states

module LSTM =
    let init (inDim: int) (hiddenDim: int) (config: LSTMConfig) (dtype: DType) (device: Device) : Result<LSTM, ToroError> =
        result {
            let! wIh = Init.toParam [ 4 * hiddenDim; inDim ] dtype device config.WIhInit

            let! wHh = Init.toParam [ 4 * hiddenDim; hiddenDim ] dtype device config.WHhInit

            let! bIh =
                config.BIhInit
                |> Option.traverseResult (Init.toParam [ 4 * hiddenDim ] dtype device)

            let! bHh =
                config.BHhInit
                |> Option.traverseResult (Init.toParam [ 4 * hiddenDim ] dtype device)

            return {
                WIh = wIh
                WHh = wHh
                BIh = bIh
                BHh = bHh
                HiddenDim = hiddenDim
                DType = dtype
                Device = device
            }
        }

    let initDefault (inDim: int) (hiddenDim: int) (dtype: DType) (device: Device) : Result<LSTM, ToroError> =
        init inDim hiddenDim LSTMConfig.defaultConfig dtype device

// --- GRU ---

type GRUState = { H: Tensor }

type GRUConfig = {
    WIhInit: Init
    WHhInit: Init
    BIhInit: Init option
    BHhInit: Init option
} with

    static member defaultConfig = {
        WIhInit = Init.KaimingNormal
        WHhInit = Init.KaimingNormal
        BIhInit = Some(Init.Const 0.0)
        BHhInit = Some(Init.Const 0.0)
    }

    static member defaultNoBias = {
        WIhInit = Init.KaimingNormal
        WHhInit = Init.KaimingNormal
        BIhInit = None
        BHhInit = None
    }

type GRU = {
    WIh: Tensor
    WHh: Tensor
    BIh: Tensor option
    BHh: Tensor option
    HiddenDim: int
    DType: DType
    Device: Device
} with

    member this.zeroState(batchDim: int) : Result<GRUState, ToroError> =
        result {
            let! zeros = Tensor.zeros ([ batchDim; this.HiddenDim ], this.DType, this.Device)

            return { H = zeros }
        }

    member this.step (input: Tensor) (state: GRUState) : Result<GRUState, ToroError> =
        result {
            let! wIhT = this.WIh.t ()
            let! wHhT = this.WHh.t ()
            let! ihGates = input.matmul wIhT
            let! hhGates = state.H.matmul wHhT

            let! ihGates =
                match this.BIh with
                | Some b -> ihGates.add b
                | None -> Ok ihGates

            let! hhGates =
                match this.BHh with
                | Some b -> hhGates.add b
                | None -> Ok hhGates

            let! chunksIh = ihGates.chunk (3, 1)
            let! chunksHh = hhGates.chunk (3, 1)

            let! rGate = (chunksIh[0] + chunksHh[0]).sigmoid ()
            let! zGate = (chunksIh[1] + chunksHh[1]).sigmoid ()

            let! rHh = rGate.mul chunksHh[2]
            let! nGate = (chunksIh[2] + rHh).tanh ()

            let! nextH = zGate.mul state.H +~ (1.0 - zGate).mul nGate

            return { H = nextH }
        }

    member this.seq(input: Tensor) : Result<GRUState list, ToroError> =
        result {
            let batchDim = input.Shape[0]
            let seqLen = input.Shape[1]
            let! initState = this.zeroState batchDim
            let mutable state = initState
            let output = System.Collections.Generic.List<GRUState>()

            for i in 0 .. seqLen - 1 do
                let step_input = input.at [ A; I i ]
                let! newState = this.step step_input state
                state <- newState
                output.Add newState

            return output |> Seq.toList
        }

    member _.statesToTensor(states: GRUState list) : Result<Tensor, ToroError> =
        let hs = states |> List.map _.H
        Tensor.stack (hs, 1)

    interface IRNN<GRUState> with
        member this.zeroState batchDim = this.zeroState batchDim
        member this.step input state = this.step input state
        member this.seq input = this.seq input
        member this.statesToTensor states = this.statesToTensor states

module GRU =
    let init (inDim: int) (hiddenDim: int) (config: GRUConfig) (dtype: DType) (device: Device) : Result<GRU, ToroError> =
        result {
            let! wIh = Init.toParam [ 3 * hiddenDim; inDim ] dtype device config.WIhInit

            let! wHh = Init.toParam [ 3 * hiddenDim; hiddenDim ] dtype device config.WHhInit

            let! bIh =
                config.BIhInit
                |> Option.traverseResult (Init.toParam [ 3 * hiddenDim ] dtype device)

            let! bHh =
                config.BHhInit
                |> Option.traverseResult (Init.toParam [ 3 * hiddenDim ] dtype device)

            return {
                WIh = wIh
                WHh = wHh
                BIh = bIh
                BHh = bHh
                HiddenDim = hiddenDim
                DType = dtype
                Device = device
            }
        }

    let initDefault (inDim: int) (hiddenDim: int) (dtype: DType) (device: Device) : Result<GRU, ToroError> =
        init inDim hiddenDim GRUConfig.defaultConfig dtype device
