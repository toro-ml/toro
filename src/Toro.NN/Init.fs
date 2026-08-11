namespace Toro.NN

open TorchSharp
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
        let initInPlace f (shape: int64[]) (dtype: torch.ScalarType) (device: torch.Device) : Tensor =
            let t = torch.empty (shape, dtype = dtype, device = device)
            f t |> ignore
            t

    let toTensor (shape: int64[]) (dtype: torch.ScalarType) (device: torch.Device) (init: Init) : Tensor =
        match init with
        | Const v -> torch.full (shape, scalar v, dtype = dtype, device = device)
        | Randn(mean, stdev) -> torch.randn(shape, dtype = dtype, device = device).mul(scalar stdev).add (scalar mean)
        | Uniform(lo, up) -> torch.rand(shape, dtype = dtype, device = device).mul(scalar (up - lo)).add (scalar lo)
        | KaimingNormal -> initInPlace (fun t -> torch.nn.init.kaiming_normal_ t) shape dtype device
        | KaimingUniform -> initInPlace (fun t -> torch.nn.init.kaiming_uniform_ t) shape dtype device
        | XavierNormal gain -> initInPlace (fun t -> torch.nn.init.xavier_normal_ (t, gain)) shape dtype device
        | XavierUniform gain -> initInPlace (fun t -> torch.nn.init.xavier_uniform_ (t, gain)) shape dtype device
        | Orthogonal gain -> initInPlace (fun t -> torch.nn.init.orthogonal_ (t, gain)) shape dtype device
        | TruncNormal(mean, stdev, lo, up) ->
            initInPlace (fun t -> torch.nn.init.trunc_normal_ (t, mean, stdev, a = lo, b = up)) shape dtype device

    let toParam (shape: int64[]) (dtype: torch.ScalarType) (device: torch.Device) (init: Init) : Tensor =
        let t = toTensor shape dtype device init
        t.requires_grad_ ()
