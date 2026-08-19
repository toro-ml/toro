namespace Toro.Models

open System.Collections.Generic
open Toro

module internal TensorOwner =

    let disposeDistinct (namedTensors: 'Owner -> seq<string * Tensor>) (owner: 'Owner) =
        let seen = HashSet<obj>(ReferenceEqualityComparer.Instance)

        for _, tensor in namedTensors owner do
            if seen.Add(box tensor) then
                tensor.Dispose()
