namespace Toro.NN

open Toro

/// Fully connected linear layer.
type Linear = {
    Weight: Tensor
    Bias: Tensor option
} with

    /// $y = xW^\top + b$
    member this.forward(x: Tensor) : Tensor =
        let wt = this.Weight.t ()
        let x = x.matmul wt

        match this.Bias with
        | None -> x
        | Some bias -> x.add bias

    interface IModule with
        member this.forward x = this.forward x

module Linear =
    /// Create a linear layer with bias.
    let init (inDim: int) (outDim: int) (dtype: DType) (device: Device) : Linear =
        let bound = 1.0 / sqrt (float inDim)

        let ws = Init.toParam [ outDim; inDim ] dtype device Init.defaultKaimingNormal
        let bs = Init.toParam [ outDim ] dtype device (Init.Uniform(-bound, bound))
        { Weight = ws; Bias = Some bs }

    /// Create a linear layer without bias.
    let initNoBias (inDim: int) (outDim: int) (dtype: DType) (device: Device) : Linear =
        let ws = Init.toParam [ outDim; inDim ] dtype device Init.defaultKaimingNormal
        { Weight = ws; Bias = None }
