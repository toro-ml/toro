namespace Toro.NN

open Toro

type Embedding = {
    Embeddings: Tensor
    HiddenSize: int
} with

    member this.forward(indexes: Tensor) : Result<Tensor, ToroError> =
        result {
            let finalDims = indexes.Shape @ [ this.HiddenSize ]

            let! flat = indexes.flattenAll ()

            let values = this.Embeddings[flat]

            return! values.reshape finalDims
        }

    interface IModule with
        member this.forward x = this.forward x

module Embedding =
    let init (inSize: int) (outSize: int) (dtype: DType) (device: Device) : Result<Embedding, ToroError> =
        result {
            let! embeddings = Init.toParam [ inSize; outSize ] dtype device (Init.Randn(0.0, 1.0))

            return {
                Embeddings = embeddings
                HiddenSize = outSize
            }
        }
