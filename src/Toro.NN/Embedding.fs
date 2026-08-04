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

            let! values = this.Embeddings.indexSelect (0, flat)

            return! values.reshape finalDims
        }

    interface IModule with
        member this.forward x = this.forward x

module Embedding =
    let create
        (inSize: int)
        (outSize: int)
        (vb: VarBuilder)
        : Result<Embedding, ToroError> =
        let init = Init.Randn(0.0, 1.0)

        result {
            let! embeddings =
                VarBuilder.getWithHints [ inSize; outSize ] "weight" init vb

            return {
                Embeddings = embeddings
                HiddenSize = outSize
            }
        }
