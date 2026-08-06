namespace Toro

[<AutoOpen>]
module TensorOp =

    let (+~) (a: Result<Tensor, ToroError>) (b: Result<Tensor, ToroError>) =
        result {
            let! a = a
            let! b = b
            return! a.add b
        }

    let (-~) (a: Result<Tensor, ToroError>) (b: Result<Tensor, ToroError>) =
        result {
            let! a = a
            let! b = b
            return! a.sub b
        }

    let ( *~ ) (a: Result<Tensor, ToroError>) (b: Result<Tensor, ToroError>) =
        result {
            let! a = a
            let! b = b
            return! a.mul b
        }

    let (/~) (a: Result<Tensor, ToroError>) (b: Result<Tensor, ToroError>) =
        result {
            let! a = a
            let! b = b
            return! a.div b
        }

module TensorR =

    let scale (s: float) (t: Result<Tensor, ToroError>) =
        Result.bind (fun (t: Tensor) -> t.mulScalar s) t

    let shift (s: float) (t: Result<Tensor, ToroError>) =
        Result.bind (fun (t: Tensor) -> t.addScalar s) t

    let sqr (t: Result<Tensor, ToroError>) =
        Result.bind (fun (t: Tensor) -> t.sqr ()) t

    let sqrt (t: Result<Tensor, ToroError>) =
        Result.bind (fun (t: Tensor) -> t.sqrt ()) t

    let neg (t: Result<Tensor, ToroError>) =
        Result.bind (fun (t: Tensor) -> t.neg ()) t

    let exp (t: Result<Tensor, ToroError>) =
        Result.bind (fun (t: Tensor) -> t.exp ()) t

    let log (t: Result<Tensor, ToroError>) =
        Result.bind (fun (t: Tensor) -> t.log ()) t

    let meanAll (t: Result<Tensor, ToroError>) =
        Result.bind (fun (t: Tensor) -> t.meanAll ()) t
