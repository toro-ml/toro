namespace Toro.NN

open Toro

/// A composable layer with typed input and output.
type IModule<'In, 'Out> =
    abstract forward: 'In -> 'Out

/// Shortcut for IModule<Tensor, Tensor>.
/// Most layers implement this interface directly.
type IModule =
    inherit IModule<Tensor, Tensor>
