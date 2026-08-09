namespace Toro.GNN

open Toro
open Toro.NN

/// Graph Isomorphism Network layer (Xu et al., 2019).
/// $\mathbf{x}_i' = h_\Theta\bigl((1 + \varepsilon) \cdot \mathbf{x}_i
///   + \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j\bigr)$
/// where $h_\Theta$ is a 2-layer MLP: Linear -> ReLU -> Linear.
type GINConv = {
    Eps: Tensor
    Linear1: Linear
    Linear2: Linear
} with

    member this.forward(x: Tensor, edgeIndex: Tensor) : Result<Tensor, ToroError> =
        let numNodes = x.Shape[0]
        let features = x.Shape[1]

        result {
            let src = edgeIndex[0]
            let tgt = edgeIndex[1]

            let msg = x[src]
            let! aggr = MessagePassing.aggregate Add msg tgt numNodes features

            let! epsVal = this.Eps.toFloat32Scalar ()
            let h = x * (1.0 + float epsVal) + aggr
            let! h = this.Linear1.forward h
            let! h = h.relu ()
            return! this.Linear2.forward h
        }

module GINConv =
    /// Create a GINConv with a 2-layer MLP. trainEps controls whether eps is learnable.
    let init
        (inChannels: int)
        (hiddenChannels: int)
        (outChannels: int)
        (trainEps: bool)
        (dtype: DType)
        (device: Device)
        : Result<GINConv, ToroError> =
        result {
            let! eps = Init.toTensor [ 1 ] dtype device (Init.Const 0.0)

            let! eps = if trainEps then eps.requiresGrad () else Ok eps

            let! lin1 = Linear.init inChannels hiddenChannels dtype device
            let! lin2 = Linear.init hiddenChannels outChannels dtype device

            return {
                Eps = eps
                Linear1 = lin1
                Linear2 = lin2
            }
        }
