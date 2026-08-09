namespace Toro.NN

open Toro

/// Gradient clipping utilities.
module Clip =

    /// Clip gradients by global L2 norm. Return the total norm before clipping.
    let gradNorm (maxNorm: float) (vars: Tensor list) : Result<float, ToroError> =
        result {
            let mutable totalNormSq = 0.0

            for v in vars do
                let! g = v.grad ()
                let! sq = g.mul g
                let! s = sq.sumAll ()
                let! f = s.toFloat64Scalar ()
                totalNormSq <- totalNormSq + f

            let totalNorm = sqrt totalNormSq

            if totalNorm > maxNorm then
                let scale = maxNorm / (totalNorm + 1e-6)

                do!
                    Toro.noGrad (fun () ->
                        result {
                            for v in vars do
                                let! g = v.grad ()
                                let! scaled = g.mulScalar scale
                                do! g.copyInPlace scaled
                        })

            return totalNorm
        }

    /// Clip each gradient element to [-value, +value].
    let gradValue (value: float) (vars: Tensor list) : Result<unit, ToroError> =
        Toro.noGrad (fun () ->
            result {
                for v in vars do
                    let! g = v.grad ()
                    let! clamped = g.clamp (-value, value)
                    do! g.copyInPlace clamped
            })
