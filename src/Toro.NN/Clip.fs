namespace Toro.NN

open TorchSharp
open Toro

/// Gradient clipping utilities.
module Clip =

    /// Clip gradients by global L2 norm. Return the total norm before clipping.
    let gradNorm (maxNorm: float) (vars: Tensor list) : float =
        let mutable totalNormSq = 0.0

        for v in vars do
            let g = v.grad ()
            let sq = g.mul g
            let s = sq.sum ()
            totalNormSq <- totalNormSq + s.ToDouble()

        let totalNorm = sqrt totalNormSq

        if totalNorm > maxNorm then
            let scale = maxNorm / (totalNorm + 1e-6)

            Toro.noGrad (fun () ->
                for v in vars do
                    let g = v.grad ()
                    let scaled = g * scalar scale
                    g.copyInPlace scaled)

        totalNorm

    /// Clip each gradient element to [-value, +value].
    let gradValue (value: float) (vars: Tensor list) : unit =
        Toro.noGrad (fun () ->
            for v in vars do
                let g = v.grad ()
                let clamped = g.clamp (scalar (-value), scalar value)
                g.copyInPlace clamped)
