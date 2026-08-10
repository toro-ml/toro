namespace Toro.NN

open System.IO
open Toro

/// Function record for checkpoint save/load operations.
type OptimizerOps = {
    SaveState: string -> Result<unit, ToroError>
    LoadState: string -> Result<unit, ToroError>
    LearningRate: unit -> float
    SetLearningRate: float -> unit
}

type SGD = {
    Vars: Tensor list
    mutable LearningRate: float
} with

    member this.step() =
        result {
            for v in this.Vars do
                do!
                    scoped {
                        let! g = v.grad ()
                        let! updated = v -~ g.mulScalar this.LearningRate
                        do! v.copyInPlace updated
                    }
        }

    member this.learningRate() = this.LearningRate

    member this.setLearningRate lr = this.LearningRate <- lr

    member this.zeroGrad() =
        for v in this.Vars do
            v.zeroGrad ()

    member _.saveState(_dirPath: string) : Result<unit, ToroError> = Ok()
    member _.loadState(_dirPath: string) : Result<unit, ToroError> = Ok()

    member this.toOps() : OptimizerOps = {
        SaveState = this.saveState
        LoadState = this.loadState
        LearningRate = this.learningRate
        SetLearningRate = this.setLearningRate
    }

module SGD =
    let create (lr: float) (vars: Tensor list) : SGD = { Vars = vars; LearningRate = lr }

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

type AdamW = {
    Vars: (Tensor * Tensor * Tensor) list
    mutable StepCount: int
    mutable Params: ParamsAdamW
} with

    member this.step() =
        this.StepCount <- this.StepCount + 1
        let p = this.Params
        let t = float this.StepCount

        result {
            for param, m, v in this.Vars do
                do!
                    scoped {
                        let! g = param.grad ()

                        let! mNew = m.mulScalar p.Beta1 +~ g.mulScalar (1.0 - p.Beta1)
                        do! m.copyInPlace mNew

                        let! vNew = v.mulScalar p.Beta2 +~ g.sqr () *~. (1.0 - p.Beta2)
                        do! v.copyInPlace vNew

                        let mHatScale = 1.0 / (1.0 - pown p.Beta1 (int t))
                        let vHatScale = 1.0 / (1.0 - pown p.Beta2 (int t))

                        let! mHat = mNew *~. mHatScale
                        let! vHat = vNew *~. vHatScale

                        let! updated =
                            param.mulScalar (1.0 - p.Lr * p.WeightDecay)
                            -~ (mHat /~ (vHat.sqrt () +~. p.Eps) *~. p.Lr)

                        do! param.copyInPlace updated
                    }
        }

    member this.learningRate() = this.Params.Lr

    member this.setLearningRate lr =
        this.Params <- { this.Params with Lr = lr }

    member this.zeroGrad() =
        for param, _, _ in this.Vars do
            param.zeroGrad ()

    member this.saveState(dirPath: string) =
        result {
            do! ToroError.wrap (fun () -> Directory.CreateDirectory dirPath |> ignore)

            let stepTensor =
                Tensor.ofList ([ int64 this.StepCount ], Cpu)
                |> Result.defaultWith (fun _ -> failwith "unreachable")

            let mutable tensors = Map [ "step", stepTensor ]

            for i, (_, m, v) in this.Vars |> List.indexed do
                tensors <- tensors |> Map.add $"m.{i}" m
                tensors <- tensors |> Map.add $"v.{i}" v

            do! SafeTensors.save tensors (Path.Combine(dirPath, "optimizer.safetensors"))
        }

    member this.loadState(dirPath: string) =
        result {
            let path = Path.Combine(dirPath, "optimizer.safetensors")

            if File.Exists path then
                do!
                    scoped {
                        let! tensors = SafeTensors.load path

                        match tensors |> Map.tryFind "step" with
                        | Some stepTensor ->
                            let! stepVal = stepTensor.toInt64Scalar ()
                            this.StepCount <- int stepVal
                        | None -> ()

                        for i, (_, m, v) in this.Vars |> List.indexed do
                            match tensors |> Map.tryFind $"m.{i}" with
                            | Some loaded -> do! m.copyInPlace loaded
                            | None -> ()

                            match tensors |> Map.tryFind $"v.{i}" with
                            | Some loaded -> do! v.copyInPlace loaded
                            | None -> ()
                    }
        }

    member this.toOps() : OptimizerOps = {
        SaveState = this.saveState
        LoadState = this.loadState
        LearningRate = this.learningRate
        SetLearningRate = this.setLearningRate
    }

module AdamW =
    let create (config: ParamsAdamW) (vars: Tensor list) : Result<AdamW, ToroError> =
        result {
            let! varTriples =
                vars
                |> List.traverseResult (fun param ->
                    result {
                        let! m = Tensor.zeros (param.Shape, param.DType, param.Device)
                        let! v = Tensor.zeros (param.Shape, param.DType, param.Device)
                        return param, m, v
                    })

            return {
                Vars = varTriples
                StepCount = 0
                Params = config
            }
        }

    let createWithLr (lr: float) (vars: Tensor list) : Result<AdamW, ToroError> =
        create
            {
                ParamsAdamW.defaultParams with
                    Lr = lr
            }
            vars
