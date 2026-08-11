namespace Toro.NN

open Toro

/// Classification metrics computed on tensors.
module Metrics =

    /// Fraction of matching elements between predictions and targets.
    /// Both tensors must have the same shape (class indices, not logits).
    let accuracy (pred: Tensor) (target: Tensor) : float =
        let correct = pred.eq target
        let correctF = correct.toDType F32
        let total = correctF.meanAll ()
        total.toFloat64Scalar ()

    /// Accuracy from logits: take argmax over the last dimension, then compare to target indices.
    let accuracyFromLogits (logits: Tensor) (target: Tensor) : float =
        let pred = logits.argmax (-1)
        accuracy pred target

    let private countForClass (pred: Tensor) (target: Tensor) (c: int) =
        let pMask = pred.eqScalar (float c)
        let tMask = target.eqScalar (float c)
        let tpMask = pMask * tMask

        let tpF = tpMask.toDType F32
        let tpSum = tpF.sumAll ()
        let tp = tpSum.toFloat64Scalar ()

        let pF = pMask.toDType F32
        let pSum = pF.sumAll ()
        let predCount = pSum.toFloat64Scalar ()

        let tF = tMask.toDType F32
        let tSum = tF.sumAll ()
        let targetCount = tSum.toFloat64Scalar ()

        tp, predCount, targetCount

    /// Per-class precision: TP / (TP + FP). Returns NaN for classes with no predicted positives.
    let precision (numClasses: int) (pred: Tensor) (target: Tensor) : float list =
        [ 0 .. numClasses - 1 ]
        |> List.map (fun c ->
            let tp, predCount, _ = countForClass pred target c
            if predCount = 0.0 then nan else tp / predCount)

    /// Per-class recall: TP / (TP + FN). Returns NaN for classes with no actual positives.
    let recall (numClasses: int) (pred: Tensor) (target: Tensor) : float list =
        [ 0 .. numClasses - 1 ]
        |> List.map (fun c ->
            let tp, _, targetCount = countForClass pred target c
            if targetCount = 0.0 then nan else tp / targetCount)

    /// Per-class F1 score: $2 \cdot \text{precision} \cdot \text{recall} / (\text{precision} + \text{recall})$.
    let f1 (numClasses: int) (pred: Tensor) (target: Tensor) : float list =
        let p = precision numClasses pred target
        let r = recall numClasses pred target

        List.map2
            (fun pi ri ->
                if
                    System.Double.IsNaN pi
                    || System.Double.IsNaN ri
                    || pi + ri = 0.0
                then
                    nan
                else
                    2.0 * pi * ri / (pi + ri))
            p
            r
