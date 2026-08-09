namespace Toro.NN

open Toro

/// Loss functions. Each takes (input, target) and returns a scalar loss tensor.
module Loss =

    /// $\text{MSE} = (1/n)\sum(x_i - y_i)^2$
    let mse (inp: Tensor) (target: Tensor) : Result<Tensor, ToroError> =
        inp.sub target |> TensorR.sqr |> TensorR.meanAll

    /// $\text{NLL} = -(1/n)\sum x_{i,y_i}$
    let nll (inp: Tensor) (target: Tensor) : Result<Tensor, ToroError> =
        result {
            let! bSz = inp.dim 0
            let! gathered = inp.gather (1, target)
            let! squeezed = gathered.squeeze -1
            let! total = (-squeezed).sumAll ()
            return! total.affine (1.0 / float bSz, 0.0)
        }

    /// $H(p,q) = -(1/n)\sum \log\text{softmax}(x)_{y_i}$
    let crossEntropy (inp: Tensor) (target: Tensor) : Result<Tensor, ToroError> =
        result {
            let! logSm = inp.logSoftmax -1
            let! target' = target.unsqueeze -1
            return! nll logSm target'
        }

    /// $\text{BCE} = \max(x,0) - x \cdot y + \ln(1 + e^{-\lvert x \rvert})$
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

    /// $\text{L1} = (1/n)\sum|x_i - y_i|$
    let l1 (inp: Tensor) (target: Tensor) : Result<Tensor, ToroError> =
        result {
            let! diff = inp.sub target
            let! absDiff = diff.abs ()
            return! absDiff.meanAll ()
        }

    /// $\text{SmoothL1} = (1/n)\sum z_i$ where $z_i = 0.5 x_i^2/\beta$ if $|x_i| < \beta$, else $|x_i| - 0.5\beta$
    let smoothL1 (beta: float) (inp: Tensor) (target: Tensor) : Result<Tensor, ToroError> =
        result {
            let! diff = inp.sub target
            let! absDiff = diff.abs ()
            let! sq = diff.mul diff
            let! sqTerm = sq.mulScalar (0.5 / beta)
            let! linTerm = absDiff.addScalar (-0.5 * beta)
            let mask = absDiff.ltScalar beta
            let! loss = Tensor.where (mask, sqTerm, linTerm)
            return! loss.meanAll ()
        }

    /// $\text{KL}(p \| q) = (1/n)\sum p_i (\log p_i - q_i)$. Expects log-probabilities as input and probabilities as target.
    let klDiv (inp: Tensor) (target: Tensor) : Result<Tensor, ToroError> =
        result {
            let! logTarget = target.log ()
            let! diff = logTarget -~ inp
            let! weighted = target.mul diff
            return! weighted.meanAll ()
        }
