namespace Toro.NN

open TorchSharp
open Toro

type Embedding = {
    Embeddings: Tensor
    HiddenSize: int64
} with

    member this.forward(indexes: Tensor) : Tensor =
        let finalDims = Array.append indexes.shape [| int64 this.HiddenSize |]
        let flat = indexes.flatten (0L, -1L)
        let values = this.Embeddings[flat]
        values.reshape finalDims

    interface IModule with
        member this.forward x = this.forward x

module Embedding =
    let init (inSize: int64) (outSize: int64) (dtype: torch.ScalarType) (device: torch.Device) : Embedding =
        let embeddings =
            Init.toParam [| inSize; outSize |] dtype device (Init.Randn(0.0, 1.0))

        {
            Embeddings = embeddings
            HiddenSize = outSize
        }
