namespace Toro

/// Result-returning arithmetic operators (<c>+~</c>, <c>-~</c>, <c>*~</c>, <c>/~</c>) and scalar variants (<c>*~.</c> etc.).
[<AutoOpen>]
module TensorOp =

    [<System.ComponentModel.EditorBrowsable(System.ComponentModel.EditorBrowsableState.Never)>]
    type ToR = ToR
        with

            static member (%%)(ToR, t: Tensor) : Result<Tensor, ToroError> = Ok t
            static member (%%)(ToR, r: Result<Tensor, ToroError>) = r

    let inline internal toR x = ToR %% x

    /// $a + b$
    let inline (+~) a b =
        result {
            let! (a: Tensor) = toR a
            let! (b: Tensor) = toR b
            return! a.add b
        }

    /// $a - b$
    let inline (-~) a b =
        result {
            let! (a: Tensor) = toR a
            let! (b: Tensor) = toR b
            return! a.sub b
        }

    /// $a \times b$
    let inline ( *~ ) a b =
        result {
            let! (a: Tensor) = toR a
            let! (b: Tensor) = toR b
            return! a.mul b
        }

    /// $a / b$
    let inline (/~) a b =
        result {
            let! (a: Tensor) = toR a
            let! (b: Tensor) = toR b
            return! a.div b
        }

    /// $t \times s$ (scalar)
    let inline ( *~. ) t (s: float) =
        result {
            let! (t: Tensor) = toR t
            return! t.mulScalar s
        }

    /// $t / s$ (scalar)
    let inline (/~.) t (s: float) =
        result {
            let! (t: Tensor) = toR t
            return! t.divScalar s
        }

    /// $t + s$ (scalar)
    let inline (+~.) t (s: float) =
        result {
            let! (t: Tensor) = toR t
            return! t.addScalar s
        }

    /// $t - s$ (scalar)
    let inline (-~.) t (s: float) =
        result {
            let! (t: Tensor) = toR t
            return! t.subScalar s
        }

/// Pipeable Result-returning tensor functions (e.g. <c>tensor |&gt; TensorR.sqrt</c>).
module TensorR =

    /// $s \cdot t$
    let inline scale (s: float) t =
        result {
            let! (t: Tensor) = toR t
            return! t.mulScalar s
        }

    /// $t + s$
    let inline shift (s: float) t =
        result {
            let! (t: Tensor) = toR t
            return! t.addScalar s
        }

    /// $t^2$
    let inline sqr t =
        result {
            let! (t: Tensor) = toR t
            return! t.sqr ()
        }

    /// $\sqrt{t}$
    let inline sqrt t =
        result {
            let! (t: Tensor) = toR t
            return! t.sqrt ()
        }

    /// $-t$
    let inline neg t =
        result {
            let! (t: Tensor) = toR t
            return! t.neg ()
        }

    /// $e^t$
    let inline exp t =
        result {
            let! (t: Tensor) = toR t
            return! t.exp ()
        }

    /// $\ln t$
    let inline log t =
        result {
            let! (t: Tensor) = toR t
            return! t.log ()
        }

    /// Mean of all elements.
    let inline meanAll t =
        result {
            let! (t: Tensor) = toR t
            return! t.meanAll ()
        }
