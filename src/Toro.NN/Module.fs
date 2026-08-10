namespace Toro.NN

open Toro

/// A composable layer. Kleisli-composable via the >=> operator and the pipeline CE.
type IModule<'In, 'Out> =
    abstract forward: 'In -> Result<'Out, ToroError>

/// Shortcut for IModule<Tensor, Tensor>.
/// Most layers implement this interface directly.
type IModule =
    inherit IModule<Tensor, Tensor>
