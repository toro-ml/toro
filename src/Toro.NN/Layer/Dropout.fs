namespace Toro.NN

open Toro

type Dropout = {
    DropP: float
} with

    member this.forwardT (train: bool) (x: Tensor) : Result<Tensor, ToroError> =
        if not train || this.DropP = 0.0 then
            Ok x
        else
            x.dropout (this.DropP, train)

module Dropout =
    let create (dropP: float) : Dropout = { DropP = dropP }
