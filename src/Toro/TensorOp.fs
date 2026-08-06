namespace Toro

[<AutoOpen>]
module TensorOp =

    [<System.ComponentModel.EditorBrowsable(System.ComponentModel.EditorBrowsableState.Never)>]
    type ToR = ToR with
        static member (%%) (ToR, t: Tensor) : Result<Tensor, ToroError> = Ok t
        static member (%%) (ToR, r: Result<Tensor, ToroError>) = r

    let inline internal toR x = ToR %% x

    let inline ( +~ ) a b =
        result {
            let! (a: Tensor) = toR a
            let! (b: Tensor) = toR b
            return! a.add b
        }

    let inline ( -~ ) a b =
        result {
            let! (a: Tensor) = toR a
            let! (b: Tensor) = toR b
            return! a.sub b
        }

    let inline ( *~ ) a b =
        result {
            let! (a: Tensor) = toR a
            let! (b: Tensor) = toR b
            return! a.mul b
        }

    let inline ( /~ ) a b =
        result {
            let! (a: Tensor) = toR a
            let! (b: Tensor) = toR b
            return! a.div b
        }

    let inline ( *~. ) t (s: float) =
        result {
            let! (t: Tensor) = toR t
            return! t.mulScalar s
        }

    let inline ( /~. ) t (s: float) =
        result {
            let! (t: Tensor) = toR t
            return! t.divScalar s
        }

    let inline ( +~. ) t (s: float) =
        result {
            let! (t: Tensor) = toR t
            return! t.addScalar s
        }

    let inline ( -~. ) t (s: float) =
        result {
            let! (t: Tensor) = toR t
            return! t.subScalar s
        }

module TensorR =

    let inline scale (s: float) t =
        result {
            let! (t: Tensor) = toR t
            return! t.mulScalar s
        }

    let inline shift (s: float) t =
        result {
            let! (t: Tensor) = toR t
            return! t.addScalar s
        }

    let inline sqr t =
        result {
            let! (t: Tensor) = toR t
            return! t.sqr ()
        }

    let inline sqrt t =
        result {
            let! (t: Tensor) = toR t
            return! t.sqrt ()
        }

    let inline neg t =
        result {
            let! (t: Tensor) = toR t
            return! t.neg ()
        }

    let inline exp t =
        result {
            let! (t: Tensor) = toR t
            return! t.exp ()
        }

    let inline log t =
        result {
            let! (t: Tensor) = toR t
            return! t.log ()
        }

    let inline meanAll t =
        result {
            let! (t: Tensor) = toR t
            return! t.meanAll ()
        }
