namespace Toro.NN

open Toro

type Linear = {
    Weight: Tensor
    Bias: Tensor option
} with

    member this.forward(x: Tensor) : Result<Tensor, ToroError> =
        result {
            let! wt = this.Weight.t ()
            let! x = x.matmul wt

            match this.Bias with
            | None -> return x
            | Some bias -> return! x.add bias
        }

    interface IModule with
        member this.forward x = this.forward x

module Linear =
    let create (inDim: int) (outDim: int) (vb: VarBuilder) : Result<Linear, ToroError> =
        let initWs = Init.defaultKaimingNormal
        let bound = 1.0 / sqrt (float inDim)

        let initBs = Init.Uniform(-bound, bound)

        result {
            let! ws = VarBuilder.getWithHints [ outDim; inDim ] "weight" initWs vb

            let! bs = VarBuilder.getWithHints [ outDim ] "bias" initBs vb

            return { Weight = ws; Bias = Some bs }
        }

    let createNoBias
        (inDim: int)
        (outDim: int)
        (vb: VarBuilder)
        : Result<Linear, ToroError> =
        let initWs = Init.defaultKaimingNormal

        result {
            let! ws = VarBuilder.getWithHints [ outDim; inDim ] "weight" initWs vb

            return { Weight = ws; Bias = None }
        }
