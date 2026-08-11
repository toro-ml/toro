namespace Toro.NN

open TorchSharp
open Toro

type Dropout = {
    DropP: float
} with

    member this.forwardT (train: bool) (x: Tensor) : Tensor =
        if not train || this.DropP = 0.0 then
            x
        else
            torch.nn.functional.dropout (x, this.DropP, train)

module Dropout =
    let create (dropP: float) : Dropout = { DropP = dropP }
