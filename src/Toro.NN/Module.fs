namespace Toro.NN

open Toro

type IModule =
    abstract forward: Tensor -> Result<Tensor, ToroError>
