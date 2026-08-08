namespace Toro.NN

open Toro

/// Loss functions. Each takes (input, target) and returns a scalar loss tensor.
module Loss =

    let mse (inp: Tensor) (target: Tensor) : Result<Tensor, ToroError> =
        inp.sub target |> TensorR.sqr |> TensorR.meanAll

    let nll (inp: Tensor) (target: Tensor) : Result<Tensor, ToroError> =
        result {
            let! bSz = inp.dim 0
            let! gathered = inp.gather (1, target)
            let! squeezed = gathered.squeeze -1
            let! total = (-squeezed).sumAll ()
            return! total.affine (1.0 / float bSz, 0.0)
        }

    let crossEntropy (inp: Tensor) (target: Tensor) : Result<Tensor, ToroError> =
        result {
            let! logSm = inp.logSoftmax -1
            let! target' = target.unsqueeze -1
            return! nll logSm target'
        }

    /// Binary cross-entropy with logits: max(x,0) - x*y + log(1 + exp(-|x|))
    let binaryCrossEntropyWithLogit (inp: Tensor) (target: Tensor) : Result<Tensor, ToroError> =
        result {
            let! loss =
                (inp.relu () -~ inp.mul target)
                +~ (inp.abs ()
                    |> TensorR.neg
                    |> TensorR.exp
                    |> TensorR.shift 1.0
                    |> TensorR.log)

            return! loss.meanAll ()
        }
