namespace Toro.NN

open System
open System.Collections.Generic
open TorchSharp
open Toro

/// Common contract for optimizers that can participate in versioned checkpoints.
type IOptimizer =
    abstract OptimizerKind: string
    abstract step: unit -> unit
    abstract zeroGrad: unit -> unit
    abstract learningRate: unit -> float
    abstract setLearningRate: float -> unit
    abstract saveState: string -> unit
    abstract validateStateDict: Map<string, Tensor> -> unit
    abstract loadStateDict: Map<string, Tensor> -> unit

module private OptimizerValidation =

    let learningRate argumentName learningRate =
        if
            Double.IsNaN learningRate
            || Double.IsInfinity learningRate
            || learningRate < 0.0
        then
            invalidArg argumentName "Learning rate must be finite and non-negative."

        learningRate

    let parameters (parameters: NamedTensor list) =
        let names = HashSet<string>(StringComparer.Ordinal)
        let tensors = HashSet<obj>(ReferenceEqualityComparer.Instance)

        for parameter in parameters do
            if String.IsNullOrWhiteSpace parameter.Name then
                invalidArg (nameof parameters) "Optimizer parameter names must not be empty."

            if parameter.Kind <> TensorKind.Parameter then
                invalidArg
                    (nameof parameters)
                    $"Optimizer input '{parameter.Name}' is a Buffer; only trainable Parameters are accepted."

            if not parameter.Tensor.requires_grad then
                invalidArg (nameof parameters) $"Optimizer input '{parameter.Name}' does not require gradients."

            if not (names.Add parameter.Name) then
                invalidArg (nameof parameters) $"Duplicate optimizer parameter name: '{parameter.Name}'."

            if not (tensors.Add(box parameter.Tensor)) then
                invalidArg
                    (nameof parameters)
                    $"Optimizer input '{parameter.Name}' shares a Tensor with another parameter. Use canonical Model.trainableParams."

        parameters

    let exactKeys optimizerKind expected (actual: Map<string, Tensor>) =
        let actualKeys = actual |> Map.keys |> Set.ofSeq
        let missing = Set.difference expected actualKeys |> Set.toList
        let unexpected = Set.difference actualKeys expected |> Set.toList

        if missing <> [] || unexpected <> [] then
            invalidOp $"{optimizerKind} optimizer state key mismatch: missing=%A{missing}; unexpected=%A{unexpected}."

    let tensor name (expected: Tensor) (actual: Tensor) =
        if expected.shape <> actual.shape then
            invalidOp $"Optimizer state '{name}' shape mismatch: expected %A{expected.shape}, got %A{actual.shape}."

        if expected.dtype <> actual.dtype then
            invalidOp $"Optimizer state '{name}' dtype mismatch: expected {expected.dtype}, got {actual.dtype}."

type SGD = private {
    Parameters: NamedTensor list
    mutable LearningRate: float
} with

    member this.step() =
        for parameter in this.Parameters do
            scoped {
                let tensor = parameter.Tensor
                let gradient = tensor.grad ()
                let updated = tensor - gradient * scalar this.LearningRate
                tensor.copyInPlace updated
            }

    member this.learningRate() = this.LearningRate

    member this.setLearningRate lr =
        this.LearningRate <- OptimizerValidation.learningRate (nameof lr) lr

    member this.zeroGrad() =
        for parameter in this.Parameters do
            parameter.Tensor.zeroGrad ()

    member _.saveState(filePath: string) = SafeTensors.save Map.empty filePath

    member _.validateStateDict(tensors: Map<string, Tensor>) =
        OptimizerValidation.exactKeys "SGD" Set.empty tensors

    member this.loadStateDict(tensors: Map<string, Tensor>) = this.validateStateDict tensors

    interface IOptimizer with
        member _.OptimizerKind = "SGD"
        member this.step() = this.step ()
        member this.zeroGrad() = this.zeroGrad ()
        member this.learningRate() = this.learningRate ()
        member this.setLearningRate lr = this.setLearningRate lr
        member this.saveState filePath = this.saveState filePath
        member this.validateStateDict tensors = this.validateStateDict tensors
        member this.loadStateDict tensors = this.loadStateDict tensors

module SGD =

    /// Create SGD over canonical, named trainable parameters.
    let create (lr: float) (parameters: NamedTensor list) : SGD = {
        Parameters = OptimizerValidation.parameters parameters
        LearningRate = OptimizerValidation.learningRate (nameof lr) lr
    }

type ParamsAdamW = {
    Lr: float
    Beta1: float
    Beta2: float
    Eps: float
    WeightDecay: float
}

module ParamsAdamW =
    let defaultParams: ParamsAdamW = {
        Lr = 0.001
        Beta1 = 0.9
        Beta2 = 0.999
        Eps = 1e-8
        WeightDecay = 0.01
    }

type private AdamWParameterState = {
    Parameter: NamedTensor
    M: Tensor
    V: Tensor
}

type AdamW = private {
    ParameterStates: AdamWParameterState list
    mutable StepCount: int
    mutable Params: ParamsAdamW
} with

    member this.step() =
        this.StepCount <- this.StepCount + 1
        let config = this.Params
        let step = this.StepCount

        for state in this.ParameterStates do
            scoped {
                let parameter = state.Parameter.Tensor
                let gradient = parameter.grad ()

                let mNew =
                    state.M * scalar config.Beta1
                    + gradient * scalar (1.0 - config.Beta1)

                state.M.copyInPlace mNew

                let vNew =
                    state.V * scalar config.Beta2
                    + gradient.square () * scalar (1.0 - config.Beta2)

                state.V.copyInPlace vNew

                let mHatScale = 1.0 / (1.0 - pown config.Beta1 step)
                let vHatScale = 1.0 / (1.0 - pown config.Beta2 step)
                let mHat = mNew * mHatScale
                let vHat = vNew * vHatScale

                let updated =
                    parameter * scalar (1.0 - config.Lr * config.WeightDecay)
                    - (mHat / (vHat.sqrt () + config.Eps) * config.Lr)

                parameter.copyInPlace updated
            }

    member this.learningRate() = this.Params.Lr

    member this.setLearningRate lr =
        this.Params <- {
            this.Params with
                Lr = OptimizerValidation.learningRate (nameof lr) lr
        }

    member this.zeroGrad() =
        for state in this.ParameterStates do
            state.Parameter.Tensor.zeroGrad ()

    member this.saveState(filePath: string) =
        use stepTensor =
            torch.tensor ([| this.StepCount |], dtype = torch.int32, device = torch.CPU)

        let tensors =
            this.ParameterStates
            |> List.fold
                (fun tensors state ->
                    tensors
                    |> Map.add $"m.{state.Parameter.Name}" state.M
                    |> Map.add $"v.{state.Parameter.Name}" state.V)
                (Map [ "step", stepTensor ])

        SafeTensors.save tensors filePath

    member this.validateStateDict(tensors: Map<string, Tensor>) =
        let expectedKeys =
            this.ParameterStates
            |> List.collect (fun state -> [ $"m.{state.Parameter.Name}"; $"v.{state.Parameter.Name}" ])
            |> Set.ofList
            |> Set.add "step"

        OptimizerValidation.exactKeys "AdamW" expectedKeys tensors

        let step = tensors["step"]

        if step.shape <> [| 1L |] then
            invalidOp $"Optimizer state 'step' shape mismatch: expected [1], got %A{step.shape}."

        if step.dtype <> torch.int32 then
            invalidOp $"Optimizer state 'step' dtype mismatch: expected Int32, got {step.dtype}."

        for state in this.ParameterStates do
            let mName = $"m.{state.Parameter.Name}"
            let vName = $"v.{state.Parameter.Name}"
            OptimizerValidation.tensor mName state.M tensors[mName]
            OptimizerValidation.tensor vName state.V tensors[vName]

    member this.loadStateDict(tensors: Map<string, Tensor>) =
        this.validateStateDict tensors

        for state in this.ParameterStates do
            state.M.copyInPlace tensors[$"m.{state.Parameter.Name}"]
            state.V.copyInPlace tensors[$"v.{state.Parameter.Name}"]

        this.StepCount <- tensors["step"].ToInt32()

    interface IOptimizer with
        member _.OptimizerKind = "AdamW"
        member this.step() = this.step ()
        member this.zeroGrad() = this.zeroGrad ()
        member this.learningRate() = this.learningRate ()
        member this.setLearningRate lr = this.setLearningRate lr
        member this.saveState filePath = this.saveState filePath
        member this.validateStateDict tensors = this.validateStateDict tensors
        member this.loadStateDict tensors = this.loadStateDict tensors

module AdamW =

    /// Create AdamW over canonical, named trainable parameters.
    let create (config: ParamsAdamW) (parameters: NamedTensor list) : AdamW =
        OptimizerValidation.learningRate "config.Lr" config.Lr
        |> ignore

        if
            config.Beta1 < 0.0
            || config.Beta1 >= 1.0
            || Double.IsNaN config.Beta1
        then
            invalidArg (nameof config) "Beta1 must be in [0, 1)."

        if
            config.Beta2 < 0.0
            || config.Beta2 >= 1.0
            || Double.IsNaN config.Beta2
        then
            invalidArg (nameof config) "Beta2 must be in [0, 1)."

        if
            config.Eps <= 0.0
            || Double.IsNaN config.Eps
            || Double.IsInfinity config.Eps
        then
            invalidArg (nameof config) "Eps must be finite and positive."

        if
            config.WeightDecay < 0.0
            || Double.IsNaN config.WeightDecay
            || Double.IsInfinity config.WeightDecay
        then
            invalidArg (nameof config) "WeightDecay must be finite and non-negative."

        let parameterStates =
            OptimizerValidation.parameters parameters
            |> List.map (fun parameter -> {
                Parameter = parameter
                M = torch.zeros_like parameter.Tensor
                V = torch.zeros_like parameter.Tensor
            })

        {
            ParameterStates = parameterStates
            StepCount = 0
            Params = config
        }

    /// Create AdamW with default parameters except for the learning rate.
    let createWithLr (lr: float) (parameters: NamedTensor list) : AdamW =
        create
            {
                ParamsAdamW.defaultParams with
                    Lr = lr
            }
            parameters
