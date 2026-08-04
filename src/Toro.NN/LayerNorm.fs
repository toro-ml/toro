namespace Toro.NN

open Toro

type LayerNormConfig = {
    Eps: float
    RemoveMean: bool
    Affine: bool
}

module LayerNormConfig =
    let defaultConfig = {
        Eps = 1e-5
        RemoveMean = true
        Affine = true
    }

type LayerNorm = {
    Weight: Tensor
    Bias: Tensor option
    RemoveMean: bool
    Eps: float
} with

    member this.forward(x: Tensor) : Result<Tensor, ToroError> =
        result {
            let xDType = x.DType

            let internalDType =
                match xDType with
                | F16
                | BF16 -> F32
                | d -> d

            let! x = x.toDType internalDType

            let! x =
                if this.RemoveMean then
                    result {
                        let! meanX = x.meanKeepdim -1
                        return! x.sub meanX
                    }
                else
                    Ok x

            let! xSqr = x.sqr ()
            let! normX = xSqr.meanKeepdim -1

            let! normXEps = normX.addScalar this.Eps

            let! normXSqrt = normXEps.sqrt ()
            let! xNormed = x.div normXSqrt
            let! xNormed = xNormed.toDType xDType

            let! x = xNormed.mul this.Weight

            match this.Bias with
            | None -> return x
            | Some bias -> return! x.add bias
        }

    interface IModule with
        member this.forward x = this.forward x

module LayerNorm =
    let create
        (size: int)
        (config: LayerNormConfig)
        (vb: VarBuilder)
        : Result<LayerNorm, ToroError> =
        result {
            let! weight = VarBuilder.getWithHints [ size ] "weight" (Init.Const 1.0) vb

            let! bias =
                if config.Affine then
                    result {
                        let! b =
                            VarBuilder.getWithHints [ size ] "bias" (Init.Const 0.0) vb

                        return Some b
                    }
                else
                    Ok None

            return {
                Weight = weight
                Bias = bias
                RemoveMean = config.RemoveMean
                Eps = config.Eps
            }
        }

    let createDefault (size: int) (vb: VarBuilder) : Result<LayerNorm, ToroError> =
        create size LayerNormConfig.defaultConfig vb

type RmsNorm = {
    Inner: LayerNorm
} with

    member this.forward(x: Tensor) : Result<Tensor, ToroError> = this.Inner.forward x

    interface IModule with
        member this.forward x = this.forward x

module RmsNorm =
    let create (size: int) (eps: float) (vb: VarBuilder) : Result<RmsNorm, ToroError> =
        let config = {
            Eps = eps
            RemoveMean = false
            Affine = false
        }

        result {
            let! inner = LayerNorm.create size config vb

            return { Inner = inner }
        }
