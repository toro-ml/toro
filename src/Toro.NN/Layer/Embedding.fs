namespace Toro.NN

open Toro

type Embedding = {
    Embeddings: Tensor
    HiddenSize: int
} with

    member this.forward(indexes: Tensor) : Tensor =
        let finalDims = indexes.Shape @ [ this.HiddenSize ]
        let flat = indexes.flattenAll ()
        let values = this.Embeddings[flat]
        values.reshape finalDims

    interface IModule with
        member this.forward x = this.forward x

module Embedding =
    let init (inSize: int) (outSize: int) (dtype: DType) (device: Device) : Embedding =
        let embeddings =
            Init.toParam [ inSize; outSize ] dtype device (Init.Randn(0.0, 1.0))

        {
            Embeddings = embeddings
            HiddenSize = outSize
        }
