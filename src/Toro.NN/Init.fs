namespace Toro.NN

open Toro

type Init =
    | Const of float
    | Randn of mean: float * stdev: float
    | Uniform of lo: float * up: float
    | KaimingNormal
    | KaimingUniform
    | XavierNormal of gain: float
    | XavierUniform of gain: float
    | Orthogonal of gain: float
    | TruncNormal of mean: float * stdev: float * lo: float * up: float

module Init =
    let defaultKaimingNormal = KaimingNormal

    [<AutoOpen>]
    module private Internal =
        let initInPlace f (shape: int list) (dtype: DType) (device: Device) : Result<Tensor, ToroError> =
            ToroError.wrap (fun () ->
                let t =
                    TorchSharp.torch.empty (
                        Shape.toInt64Array shape,
                        dtype = DType.toTorch dtype,
                        device = Device.toTorch device
                    )

                f t |> ignore
                t)
            |> Result.bind Tensor.ofTorchTensor

    let toTensor (shape: int list) (dtype: DType) (device: Device) (init: Init) : Result<Tensor, ToroError> =
        match init with
        | Const v -> Tensor.full (shape, v, dtype, device)
        | Randn(mean, stdev) -> Tensor.randn (shape, dtype, device) *~. stdev +~. mean
        | Uniform(lo, up) -> Tensor.rand (shape, dtype, device) *~. (up - lo) +~. lo
        | KaimingNormal -> initInPlace (fun t -> TorchSharp.torch.nn.init.kaiming_normal_ t) shape dtype device
        | KaimingUniform -> initInPlace (fun t -> TorchSharp.torch.nn.init.kaiming_uniform_ t) shape dtype device
        | XavierNormal gain -> initInPlace (fun t -> TorchSharp.torch.nn.init.xavier_normal_ (t, gain)) shape dtype device
        | XavierUniform gain -> initInPlace (fun t -> TorchSharp.torch.nn.init.xavier_uniform_ (t, gain)) shape dtype device
        | Orthogonal gain -> initInPlace (fun t -> TorchSharp.torch.nn.init.orthogonal_ (t, gain)) shape dtype device
        | TruncNormal(mean, stdev, lo, up) ->
            initInPlace (fun t -> TorchSharp.torch.nn.init.trunc_normal_ (t, mean, stdev, a = lo, b = up)) shape dtype device

    let toParam (shape: int list) (dtype: DType) (device: Device) (init: Init) : Result<Tensor, ToroError> =
        result {
            let! t = toTensor shape dtype device init
            return! t.requiresGrad ()
        }
