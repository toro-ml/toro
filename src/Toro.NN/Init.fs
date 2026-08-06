namespace Toro.NN

open Toro

type Init =
    | Const of float
    | Randn of mean: float * stdev: float
    | Uniform of lo: float * up: float
    | KaimingNormal

module Init =
    let defaultKaimingNormal = KaimingNormal

    let toTensor
        (shape: int list)
        (dtype: DType)
        (device: Device)
        (init: Init)
        : Result<Tensor, ToroError> =
        match init with
        | Const v -> Tensor.full (shape, v, dtype, device)
        | Randn(mean, stdev) ->
            Tensor.randn (shape, dtype, device) *~. stdev +~. mean
        | Uniform(lo, up) ->
            Tensor.rand (shape, dtype, device) *~. (up - lo) +~. lo
        | KaimingNormal ->
            let fanIn = if shape.Length >= 2 then shape[1] else shape[0]
            let stdev = sqrt (2.0 / float fanIn)
            Tensor.randn (shape, dtype, device) *~. stdev
