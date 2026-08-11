namespace Toro.NN

open TorchSharp
open Toro

/// Loss functions. Each takes (input, target) and returns a scalar loss tensor.
module Loss =

    /// $\text{MSE} = (1/n)\sum(x_i - y_i)^2$
    let mse (inp: Tensor) (target: Tensor) : Tensor =
        torch.nn.functional.mse_loss (inp, target)

    /// $\text{NLL} = -(1/n)\sum x_{i,y_i}$
    let nll (inp: Tensor) (target: Tensor) : Tensor =
        let t = if target.dim () > 1 then target.squeeze -1L else target

        torch.nn.functional.nll_loss (inp, t)

    /// $H(p,q) = -(1/n)\sum \log\text{softmax}(x)_{y_i}$
    let crossEntropy (inp: Tensor) (target: Tensor) : Tensor =
        torch.nn.functional.cross_entropy (inp, target)

    /// $\text{BCE} = \max(x,0) - x \cdot y + \ln(1 + e^{-\lvert x \rvert})$
    let binaryCrossEntropyWithLogit (inp: Tensor) (target: Tensor) : Tensor =
        torch.nn.functional.binary_cross_entropy_with_logits (inp, target)

    /// $\text{L1} = (1/n)\sum|x_i - y_i|$
    let l1 (inp: Tensor) (target: Tensor) : Tensor =
        torch.nn.functional.l1_loss (inp, target)

    /// $\text{SmoothL1} = (1/n)\sum z_i$ where $z_i = 0.5 x_i^2/\beta$ if $|x_i| < \beta$, else $|x_i| - 0.5\beta$
    let smoothL1 (beta: float) (inp: Tensor) (target: Tensor) : Tensor =
        torch.nn.functional.smooth_l1_loss (inp, target, beta = beta)

    /// $\text{KL}(p \| q) = (1/n)\sum p_i (\log p_i - q_i)$. Expects log-probabilities as input and probabilities as target.
    let klDiv (inp: Tensor) (target: Tensor) : Tensor =
        torch.nn.functional.kl_div (inp, target)

    /// Huber loss: smooth combination of L1 and L2.
    let huber (delta: float) (inp: Tensor) (target: Tensor) : Tensor =
        torch.nn.functional.huber_loss (inp, target, delta = delta)

    /// CTC loss for sequence-to-sequence alignment.
    let ctc (logProbs: Tensor) (targets: Tensor) (inputLengths: Tensor) (targetLengths: Tensor) : Tensor =
        torch.nn.functional.ctc_loss (logProbs, targets, inputLengths, targetLengths)

    /// Triplet margin loss for metric learning.
    let tripletMargin (margin: float) (anchor: Tensor) (positive: Tensor) (negative: Tensor) : Tensor =
        torch.nn.functional.triplet_margin_loss (anchor, positive, negative, margin = margin)

    /// Cosine embedding loss for similarity learning.
    let cosineEmbedding (margin: float) (x1: Tensor) (x2: Tensor) (target: Tensor) : Tensor =
        torch.nn.functional.cosine_embedding_loss (x1, x2, target, margin = margin)
