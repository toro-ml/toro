namespace Toro.Models.Interop

open System.Collections.Generic
open System.ComponentModel
open Toro

/// Tensor ownership helpers for Toro model-family packages.
[<EditorBrowsable(EditorBrowsableState.Never)>]
module TensorOwner =

    /// Dispose each distinct tensor returned by a named-tensor projection exactly once.
    let disposeDistinct (namedTensors: 'Owner -> seq<string * Tensor>) (owner: 'Owner) =
        let seen = HashSet<obj>(ReferenceEqualityComparer.Instance)

        for _, tensor in namedTensors owner do
            if seen.Add(box tensor) then
                tensor.Dispose()
