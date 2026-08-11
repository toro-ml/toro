namespace Toro.NN

open Toro

module RNN =
    /// Scan a step function over the sequence dimension, collecting all states.
    /// Input shape: [batch; seqLen; features].
    let scan (zeroState: int -> 's) (step: Tensor -> 's -> 's) (input: Tensor) : 's list =
        let seqLen = input.Shape[1]
        let s0 = zeroState (input.Shape[0])

        [ 0 .. seqLen - 1 ]
        |> List.fold
            (fun (state, revStates) i ->
                let next = step (input.at [ A; I i ]) state
                next, next :: revStates)
            (s0, [])
        |> snd
        |> List.rev

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

    member this.zeroState(batchDim: int) : LSTMState =
        let zeros = Tensor.zeros ([ batchDim; this.HiddenDim ], this.DType, this.Device)
        let zeros2 = zeros.clone ()
        { H = zeros; C = zeros2 }

    member this.step (input: Tensor) (state: LSTMState) : LSTMState =
        let wIhT = this.WIh.t ()
        let wHhT = this.WHh.t ()
        let ihGates = input.matmul wIhT
        let hhGates = state.H.matmul wHhT

        let ihGates =
            match this.BIh with
            | Some b -> ihGates.add b
            | None -> ihGates

        let hhGates =
            match this.BHh with
            | Some b -> hhGates.add b
            | None -> hhGates

        let gates = ihGates.add hhGates
        let chunks = gates.chunk (4, 1)

        let inGate = chunks[0].sigmoid ()
        let forgetGate = chunks[1].sigmoid ()
        let cellGate = chunks[2].tanh ()
        let outGate = chunks[3].sigmoid ()

        let nextC = forgetGate.mul state.C + inGate.mul cellGate
        let nextH = outGate * nextC.tanh ()

        { H = nextH; C = nextC }

module LSTM =
    let init (inDim: int) (hiddenDim: int) (config: LSTMConfig) (dtype: DType) (device: Device) : LSTM =
        let wIh = Init.toParam [ 4 * hiddenDim; inDim ] dtype device config.WIhInit
        let wHh = Init.toParam [ 4 * hiddenDim; hiddenDim ] dtype device config.WHhInit

        let bIh =
            config.BIhInit
            |> Option.map (Init.toParam [ 4 * hiddenDim ] dtype device)

        let bHh =
            config.BHhInit
            |> Option.map (Init.toParam [ 4 * hiddenDim ] dtype device)

        {
            WIh = wIh
            WHh = wHh
            BIh = bIh
            BHh = bHh
            HiddenDim = hiddenDim
            DType = dtype
            Device = device
        }

    let initDefault (inDim: int) (hiddenDim: int) (dtype: DType) (device: Device) : LSTM =
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

    member this.zeroState(batchDim: int) : GRUState =
        let zeros = Tensor.zeros ([ batchDim; this.HiddenDim ], this.DType, this.Device)
        { H = zeros }

    member this.step (input: Tensor) (state: GRUState) : GRUState =
        let wIhT = this.WIh.t ()
        let wHhT = this.WHh.t ()
        let ihGates = input.matmul wIhT
        let hhGates = state.H.matmul wHhT

        let ihGates =
            match this.BIh with
            | Some b -> ihGates.add b
            | None -> ihGates

        let hhGates =
            match this.BHh with
            | Some b -> hhGates.add b
            | None -> hhGates

        let chunksIh = ihGates.chunk (3, 1)
        let chunksHh = hhGates.chunk (3, 1)

        let rGate = (chunksIh[0] + chunksHh[0]).sigmoid ()
        let zGate = (chunksIh[1] + chunksHh[1]).sigmoid ()

        let rHh = rGate.mul chunksHh[2]
        let nGate = (chunksIh[2] + rHh).tanh ()

        let nextH = zGate.mul state.H + (1.0 - zGate).mul nGate

        { H = nextH }

module GRU =
    let init (inDim: int) (hiddenDim: int) (config: GRUConfig) (dtype: DType) (device: Device) : GRU =
        let wIh = Init.toParam [ 3 * hiddenDim; inDim ] dtype device config.WIhInit
        let wHh = Init.toParam [ 3 * hiddenDim; hiddenDim ] dtype device config.WHhInit

        let bIh =
            config.BIhInit
            |> Option.map (Init.toParam [ 3 * hiddenDim ] dtype device)

        let bHh =
            config.BHhInit
            |> Option.map (Init.toParam [ 3 * hiddenDim ] dtype device)

        {
            WIh = wIh
            WHh = wHh
            BIh = bIh
            BHh = bHh
            HiddenDim = hiddenDim
            DType = dtype
            Device = device
        }

    let initDefault (inDim: int) (hiddenDim: int) (dtype: DType) (device: Device) : GRU =
        init inDim hiddenDim GRUConfig.defaultConfig dtype device
