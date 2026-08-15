namespace Toro.NN

open TorchSharp
open Toro

/// Gradient clipping utilities.
module Clip =

    /// Clip gradients by global L2 norm. Return the total norm before clipping.
    let gradNorm (maxNorm: float) (vars: Tensor list) : float =
        let totalNormSq =
            vars
            |> List.sumBy (fun tensor ->
                scoped {
                    let gradient = tensor.grad ()
                    let squared = gradient.square ()
                    return squared.sum().ToDouble()
                })

        let totalNorm = sqrt totalNormSq

        if totalNorm > maxNorm then
            let scale = maxNorm / (totalNorm + 1e-6)

            Toro.noGrad (fun () ->
                vars
                |> List.iter (fun tensor ->
                    scoped {
                        let gradient = tensor.grad ()
                        let scaled = gradient * scalar scale
                        gradient.copyInPlace scaled
                    }))

        totalNorm

    /// Clip each gradient element to [-value, +value].
    let gradValue (value: float) (vars: Tensor list) : unit =
        Toro.noGrad (fun () ->
            vars
            |> List.iter (fun tensor ->
                scoped {
                    let gradient = tensor.grad ()
                    let clamped = gradient.clamp (scalar (-value), scalar value)
                    gradient.copyInPlace clamped
                }))
