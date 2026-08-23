namespace Toro.NN

open TorchSharp
open Toro

/// Pooling over variable-length token sequences.
module SequencePool =

    /// Average `hidden` over the sequence dimension using `mask`.
    /// `hidden` has shape [batch, sequence, dim]; `mask` has shape [batch, sequence].
    let maskedMean (hidden: Tensor) (mask: Tensor) : Tensor =
        let mask = mask.to_type(hidden.dtype).unsqueeze -1L
        let summed = hidden.mul(mask).sum ([| 1L |])
        let counts = mask.sum([| 1L |]).clamp_min 1e-9
        summed.div counts
