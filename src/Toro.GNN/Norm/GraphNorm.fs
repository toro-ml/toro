namespace Toro.GNN

open Toro
open Toro.NN

/// GraphNorm layer (Cai et al., 2021).
/// $\mathbf{x}_i' = \frac{\mathbf{x}_i - \alpha \odot E[\mathbf{x}]}
///   {\sqrt{\text{Var}[\mathbf{x}_i - \alpha \odot E[\mathbf{x}]] + \epsilon}}
///   \odot \gamma + \beta$
type GraphNorm = {
    Gamma: Tensor
    Beta: Tensor
    Alpha: Tensor
    Eps: float
} with

    /// Normalize node features per graph.
    /// x: [N, F], batch: [N] option (None treats all nodes as one graph).
    member this.forward(x: Tensor, batch: Tensor option) : Tensor =
        let numNodes = x.Shape[0]

        let batch =
            match batch with
            | Some b -> b
            | None -> Tensor.zeros ([ int numNodes ], I64, x.Device)

        let numGraphs = int (batch.Inner.max().item<int64> ()) + 1

        let mean = GlobalPool.globalMeanPool x batch numGraphs
        let alphaMean = this.Alpha * mean[batch]
        let centered = x - alphaMean

        let centeredSq = centered.mul centered
        let variance = GlobalPool.globalMeanPool centeredSq batch numGraphs

        let stddev = (variance[batch] + this.Eps).sqrt ()
        let normalized = centered.div stddev

        normalized * this.Gamma + this.Beta

module GraphNorm =
    /// Create a GraphNorm layer for features of size numFeatures.
    let init (numFeatures: int) (dtype: DType) (device: Device) : GraphNorm =
        let gamma = Init.toParam [ numFeatures ] dtype device (Init.Const 1.0)
        let beta = Init.toParam [ numFeatures ] dtype device (Init.Const 0.0)
        let alpha = Init.toParam [ numFeatures ] dtype device (Init.Const 1.0)

        {
            Gamma = gamma
            Beta = beta
            Alpha = alpha
            Eps = 1e-5
        }
