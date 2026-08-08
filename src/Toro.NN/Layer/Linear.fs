namespace Toro.NN

open Toro

/// Fully connected linear layer.
type Linear = {
    Weight: Tensor
    Bias: Tensor option
} with

    /// $y = xW^\top + b$
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
    /// Create a linear layer with bias.
    let init (inDim: int) (outDim: int) (dtype: DType) (device: Device) : Result<Linear, ToroError> =
        let bound = 1.0 / sqrt (float inDim)

        result {
            let! ws = Init.toParam [ outDim; inDim ] dtype device Init.defaultKaimingNormal

            let! bs = Init.toParam [ outDim ] dtype device (Init.Uniform(-bound, bound))
            return { Weight = ws; Bias = Some bs }
        }

    /// Create a linear layer without bias.
    let initNoBias (inDim: int) (outDim: int) (dtype: DType) (device: Device) : Result<Linear, ToroError> =
        result {
            let! ws = Init.toParam [ outDim; inDim ] dtype device Init.defaultKaimingNormal

            return { Weight = ws; Bias = None }
        }
