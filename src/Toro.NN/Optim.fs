namespace Toro.NN

open Toro

type IOptimizer =
    abstract step: unit -> Result<unit, ToroError>
    abstract backwardStep: Tensor -> Result<unit, ToroError>
    abstract learningRate: unit -> float
    abstract setLearningRate: float -> unit
    abstract zeroGrad: unit -> unit

type SGD = {
    Vars: Tensor list
    mutable LearningRate: float
} with

    interface IOptimizer with
        member this.step() =
            result {
                for v in this.Vars do
                    let! g = v.grad ()
                    let! updated = v -~ g.mulScalar this.LearningRate
                    do! v.copyInPlace updated
            }

        member this.backwardStep loss =
            let opt = this :> IOptimizer
            opt.zeroGrad ()

            result {
                do! loss.backward ()
                do! opt.step ()
            }

        member this.learningRate() = this.LearningRate

        member this.setLearningRate lr = this.LearningRate <- lr

        member this.zeroGrad() =
            for v in this.Vars do
                v.zeroGrad ()

module SGD =
    let create (lr: float) (vars: Tensor list) : SGD = {
        Vars = vars
        LearningRate = lr
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

type AdamW = {
    Vars: (Tensor * Tensor * Tensor) list
    mutable StepCount: int
    mutable Params: ParamsAdamW
} with

    interface IOptimizer with
        member this.step() =
            this.StepCount <- this.StepCount + 1
            let p = this.Params
            let t = float this.StepCount

            result {
                for (param, m, v) in this.Vars do
                    let! g = param.grad ()

                    // m = beta1 * m + (1 - beta1) * g
                    let! mNew = m.mulScalar p.Beta1 +~ g.mulScalar (1.0 - p.Beta1)
                    do! m.copyInPlace mNew

                    // v = beta2 * v + (1 - beta2) * g^2
                    let! vNew = v.mulScalar p.Beta2 +~ g.sqr () *~. (1.0 - p.Beta2)
                    do! v.copyInPlace vNew

                    // bias correction
                    let mHatScale = 1.0 / (1.0 - pown p.Beta1 (int t))
                    let vHatScale = 1.0 / (1.0 - pown p.Beta2 (int t))

                    let! mHat = mNew.mulScalar mHatScale
                    let! vHat = vNew.mulScalar vHatScale

                    // theta = theta * (1 - lr * wd) - lr * mHat / (sqrt(vHat) + eps)
                    let! updated =
                        param.mulScalar (1.0 - p.Lr * p.WeightDecay)
                        -~ (mHat /~ (vHat.sqrt () +~. p.Eps) *~. p.Lr)

                    do! param.copyInPlace updated
            }

        member this.backwardStep loss =
            let opt = this :> IOptimizer
            opt.zeroGrad ()

            result {
                do! loss.backward ()
                do! opt.step ()
            }

        member this.learningRate() = this.Params.Lr

        member this.setLearningRate lr =
            this.Params <- { this.Params with Lr = lr }

        member this.zeroGrad() =
            for (param, _, _) in this.Vars do
                param.zeroGrad ()

module AdamW =
    let create (config: ParamsAdamW) (vars: Tensor list) : Result<AdamW, ToroError> =
        result {
            let! varTriples =
                vars
                |> List.map (fun param ->
                    result {
                        let! m = Tensor.zeros (param.Shape, param.DType, param.Device)
                        let! v = Tensor.zeros (param.Shape, param.DType, param.Device)
                        return (param, m, v)
                    })
                |> List.fold
                    (fun acc r ->
                        result {
                            let! lst = acc
                            let! item = r
                            return lst @ [ item ]
                        })
                    (Ok [])

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
