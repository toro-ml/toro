namespace Toro.NN

open Toro

type IModule =
    abstract forward: Tensor -> Result<Tensor, ToroError>

type IModuleT =
    abstract forwardT: Tensor -> train: bool -> Result<Tensor, ToroError>

type ModuleTOfModule = {
    Module: IModule
} with

    interface IModuleT with
        member this.forwardT x _train = this.Module.forward x

module ModuleT =
    let ofModule (m: IModule) : IModuleT = { Module = m } :> IModuleT
