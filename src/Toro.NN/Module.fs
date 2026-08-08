namespace Toro.NN

open Toro

/// A module whose forward pass does not depend on train/eval mode.
type IModule =
    abstract forward: Tensor -> Result<Tensor, ToroError>

/// A module whose forward pass depends on train/eval mode (e.g. Dropout, BatchNorm).
type IModuleT =
    abstract forwardT: Tensor -> train: bool -> Result<Tensor, ToroError>

type ModuleTOfModule = {
    Module: IModule
} with

    interface IModuleT with
        member this.forwardT x _train = this.Module.forward x

module ModuleT =
    let ofModule (m: IModule) : IModuleT = { Module = m } :> IModuleT
