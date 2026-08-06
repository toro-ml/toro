namespace Toro.NN

open Toro

type IModule =
    abstract forward: Tensor -> Result<Tensor, ToroError>

type IModuleT =
    abstract forwardT: Tensor -> train: bool -> Result<Tensor, ToroError>

module ModuleT =
    let ofModule (m: IModule) : IModuleT =
        { new IModuleT with
            member _.forwardT x _train = m.forward x }
