namespace Toro.NN

open Toro

type GroupNormConfig = {
    Eps: float
    Affine: bool
}

module GroupNormConfig =
    let defaultConfig = { Eps = 1e-5; Affine = true }

type GroupNorm = {
    NumGroups: int
    Weight: Tensor option
    Bias: Tensor option
    Eps: float
} with

    member this.forward(x: Tensor) : Result<Tensor, ToroError> =
        x.groupNorm (this.NumGroups, ?weight = this.Weight, ?bias = this.Bias, eps = this.Eps)

    interface IModule with
        member this.forward x = this.forward x

module GroupNorm =
    let create
        (numGroups: int)
        (numChannels: int)
        (config: GroupNormConfig)
        (vb: VarBuilder)
        : Result<GroupNorm, ToroError> =
        result {
            let! weight, bias =
                if config.Affine then
                    result {
                        let! w = VarBuilder.getWithHints [ numChannels ] "weight" (Init.Const 1.0) vb
                        let! b = VarBuilder.getWithHints [ numChannels ] "bias" (Init.Const 0.0) vb
                        return Some w, Some b
                    }
                else
                    Ok(None, None)

            return {
                NumGroups = numGroups
                Weight = weight
                Bias = bias
                Eps = config.Eps
            }
        }

    let createDefault
        (numGroups: int)
        (numChannels: int)
        (vb: VarBuilder)
        : Result<GroupNorm, ToroError> =
        create numGroups numChannels GroupNormConfig.defaultConfig vb
