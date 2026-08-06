namespace Toro.NN

open Toro

type Dropout =
    { DropP: float }

    member this.forwardT (x: Tensor) (train: bool) : Result<Tensor, ToroError> =
        if not train || this.DropP = 0.0 then
            Ok x
        else
            x.dropout (this.DropP, train)

    interface IModuleT with
        member this.forwardT x train = this.forwardT x train

module Dropout =
    let create (dropP: float) : Dropout = { DropP = dropP }
