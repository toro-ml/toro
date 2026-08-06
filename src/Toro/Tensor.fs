namespace Toro

open TorchSharp

[<AutoOpen>]
module internal ScalarHelper =
    let toScalar (v: float) : Scalar = Scalar.op_Implicit v

type Tensor internal (inner: torch.Tensor) =

    member _.Inner = inner

    member _.Shape = inner.shape |> Shape.ofInt64Array

    member _.Rank = int inner.ndim

    member _.DType = DType.ofTorch inner.dtype

    member _.Device = Device.ofTorch inner.device

    member _.ElemCount = inner.NumberOfElements

    member _.IsContiguous = inner.is_contiguous ()

    // --- Factory methods ---

    static member zeros(shape: int list, dtype: DType, device: Device) =
        ToroError.wrap (fun () ->
            let t =
                torch.zeros (
                    Shape.toInt64Array shape,
                    dtype = DType.toTorch dtype,
                    device = Device.toTorch device
                )

            Tensor(t))

    static member ones(shape: int list, dtype: DType, device: Device) =
        ToroError.wrap (fun () ->
            let t =
                torch.ones (
                    Shape.toInt64Array shape,
                    dtype = DType.toTorch dtype,
                    device = Device.toTorch device
                )

            Tensor(t))

    static member full(shape: int list, value: float, dtype: DType, device: Device) =
        ToroError.wrap (fun () ->
            let t =
                torch.full (
                    Shape.toInt64Array shape,
                    toScalar value,
                    dtype = DType.toTorch dtype,
                    device = Device.toTorch device
                )

            Tensor(t))

    static member rand(shape: int list, dtype: DType, device: Device) =
        ToroError.wrap (fun () ->
            let t =
                torch.rand (
                    Shape.toInt64Array shape,
                    dtype = DType.toTorch dtype,
                    device = Device.toTorch device
                )

            Tensor(t))

    static member randn(shape: int list, dtype: DType, device: Device) =
        ToroError.wrap (fun () ->
            let t =
                torch.randn (
                    Shape.toInt64Array shape,
                    dtype = DType.toTorch dtype,
                    device = Device.toTorch device
                )

            Tensor(t))

    static member arange(stop: float, dtype: DType, device: Device) =
        ToroError.wrap (fun () ->
            let t =
                torch.arange (
                    toScalar stop,
                    dtype = DType.toTorch dtype,
                    device = Device.toTorch device
                )

            Tensor(t))

    static member arange(start: float, stop: float, dtype: DType, device: Device) =
        ToroError.wrap (fun () ->
            let t =
                torch.arange (
                    toScalar start,
                    toScalar stop,
                    dtype = DType.toTorch dtype,
                    device = Device.toTorch device
                )

            Tensor(t))

    static member ofFloat32Array2D(data: float32 array array, device: Device) =
        ToroError.wrap (fun () ->
            let rows = data.Length
            let cols = data[0].Length
            let flat = Array.concat data

            let t =
                torch
                    .tensor(flat, device = Device.toTorch device)
                    .reshape ([| int64 rows; int64 cols |])

            Tensor(t))

    static member ofFloat32Array(data: float32 array, device: Device) =
        ToroError.wrap (fun () ->
            let t = torch.tensor (data, device = Device.toTorch device)

            Tensor(t))

    static member cat(tensors: Tensor list, dim: int) =
        ToroError.wrap (fun () ->
            let ts = tensors |> List.toArray |> Array.map (fun t -> t.Inner)

            Tensor(torch.cat (ts, int64 dim)))

    static member stack(tensors: Tensor list, dim: int) =
        ToroError.wrap (fun () ->
            let ts = tensors |> List.toArray |> Array.map (fun t -> t.Inner)

            Tensor(torch.stack (ts, int64 dim)))

    // --- Dimension query ---

    member t.dim(d: int) =
        ToroError.wrap (fun () -> int (inner.size (d)))

    // --- Arithmetic (tensor-tensor) ---

    member _.add(other: Tensor) =
        ToroError.wrap (fun () -> Tensor(inner.add (other.Inner)))

    member _.sub(other: Tensor) =
        ToroError.wrap (fun () -> Tensor(inner.sub (other.Inner)))

    member _.mul(other: Tensor) =
        ToroError.wrap (fun () -> Tensor(inner.mul (other.Inner)))

    member _.div(other: Tensor) =
        ToroError.wrap (fun () -> Tensor(inner.div (other.Inner)))

    // --- Arithmetic (scalar) ---

    member _.addScalar(s: float) =
        ToroError.wrap (fun () -> Tensor(inner.add (toScalar s: Scalar)))

    member _.mulScalar(s: float) =
        ToroError.wrap (fun () -> Tensor(inner.mul (toScalar s: Scalar)))

    member _.subScalar(s: float) =
        ToroError.wrap (fun () -> Tensor(inner.sub (toScalar s: Scalar)))

    member _.divScalar(s: float) =
        ToroError.wrap (fun () -> Tensor(inner.div (toScalar s: Scalar)))

    // --- Matrix ops ---

    member _.matmul(other: Tensor) =
        ToroError.wrap (fun () -> Tensor(inner.matmul (other.Inner)))

    member _.t() =
        ToroError.wrap (fun () -> Tensor(inner.t ()))

    member _.transpose(dim0: int, dim1: int) =
        ToroError.wrap (fun () -> Tensor(inner.transpose (int64 dim0, int64 dim1)))

    // --- Shape manipulation ---

    member _.reshape(shape: int list) =
        ToroError.wrap (fun () -> Tensor(inner.reshape (Shape.toInt64Array shape)))

    member _.view(shape: int list) =
        ToroError.wrap (fun () -> Tensor(inner.view (Shape.toInt64Array shape)))

    member _.flatten(startDim: int, endDim: int) =
        ToroError.wrap (fun () -> Tensor(inner.flatten (int64 startDim, int64 endDim)))

    member _.flattenAll() =
        ToroError.wrap (fun () -> Tensor(inner.flatten (0L, -1L)))

    member _.squeeze(dim: int) =
        ToroError.wrap (fun () -> Tensor(inner.squeeze (int64 dim)))

    member _.unsqueeze(dim: int) =
        ToroError.wrap (fun () -> Tensor(inner.unsqueeze (int64 dim)))

    member _.contiguous() =
        ToroError.wrap (fun () -> Tensor(inner.contiguous ()))

    member _.broadcastLeft(shape: int list) =
        ToroError.wrap (fun () -> Tensor(inner.broadcast_to (Shape.toInt64Array shape)))

    // --- Reduction ---

    member _.sumAll() =
        ToroError.wrap (fun () -> Tensor(inner.sum ()))

    member _.sum(dim: int, ?keepDim: bool) =
        let kd = defaultArg keepDim false

        ToroError.wrap (fun () -> Tensor(inner.sum ([| int64 dim |], keepdim = kd)))

    member _.sumKeepdim(dim: int) =
        ToroError.wrap (fun () -> Tensor(inner.sum ([| int64 dim |], keepdim = true)))

    member _.meanAll() =
        ToroError.wrap (fun () -> Tensor(inner.mean ()))

    member _.mean(dim: int, ?keepDim: bool) =
        let kd = defaultArg keepDim false

        ToroError.wrap (fun () -> Tensor(inner.mean ([| int64 dim |], keepdim = kd)))

    member _.meanKeepdim(dim: int) =
        ToroError.wrap (fun () -> Tensor(inner.mean ([| int64 dim |], keepdim = true)))

    // --- Unary ops ---

    member _.neg() =
        ToroError.wrap (fun () -> Tensor(inner.neg ()))

    member _.abs() =
        ToroError.wrap (fun () -> Tensor(inner.abs ()))

    member _.sqrt() =
        ToroError.wrap (fun () -> Tensor(inner.sqrt ()))

    member _.sqr() =
        ToroError.wrap (fun () -> Tensor(inner.square ()))

    member _.pow(exponent: float) =
        ToroError.wrap (fun () -> Tensor(inner.pow (toScalar exponent)))

    member _.exp() =
        ToroError.wrap (fun () -> Tensor(inner.exp ()))

    member _.log() =
        ToroError.wrap (fun () -> Tensor(inner.log ()))

    member _.relu() =
        ToroError.wrap (fun () -> Tensor(inner.relu ()))

    member _.gelu() =
        ToroError.wrap (fun () -> Tensor(torch.nn.functional.gelu (inner)))

    member _.silu() =
        ToroError.wrap (fun () -> Tensor(torch.nn.functional.silu (inner)))

    member _.tanh() =
        ToroError.wrap (fun () -> Tensor(inner.tanh ()))

    member _.sigmoid() =
        ToroError.wrap (fun () -> Tensor(inner.sigmoid ()))

    member _.softmax(dim: int) =
        ToroError.wrap (fun () -> Tensor(torch.nn.functional.softmax (inner, int64 dim)))

    member _.logSoftmax(dim: int) =
        ToroError.wrap (fun () -> Tensor(torch.nn.functional.log_softmax (inner, int64 dim)))

    member _.clamp(min: float, max: float) =
        ToroError.wrap (fun () -> Tensor(inner.clamp (toScalar min, toScalar max)))

    member _.affine(mul: float, add: float) =
        ToroError.wrap (fun () ->
            let t = inner.mul (toScalar mul: Scalar)
            Tensor(t.add (toScalar add: Scalar)))

    // --- Indexing ---

    member _.indexSelect(dim: int, index: Tensor) =
        ToroError.wrap (fun () -> Tensor(inner.index_select (int64 dim, index.Inner)))

    member _.gather(dim: int, index: Tensor) =
        ToroError.wrap (fun () -> Tensor(inner.gather (int64 dim, index.Inner)))

    // --- Type / Device conversion ---

    member _.toDevice(device: Device) =
        ToroError.wrap (fun () -> Tensor(inner.``to`` (Device.toTorch device)))

    member _.toDType(dtype: DType) =
        ToroError.wrap (fun () -> Tensor(inner.``to`` (DType.toTorch dtype)))

    // --- Autograd ---

    member _.requiresGrad(?requiresGrad: bool) =
        let rg = defaultArg requiresGrad true

        ToroError.wrap (fun () -> Tensor(inner.requires_grad_ (rg)))

    member _.backward() =
        ToroError.wrap (fun () -> inner.backward ())

    member _.grad() =
        ToroError.wrap (fun () -> Tensor(inner.grad))

    member _.detach() =
        ToroError.wrap (fun () -> Tensor(inner.detach ()))

    member _.zeroGrad() =
        if not (isNull inner.grad) then
            inner.grad.zero_ () |> ignore

    member _.copyInPlace(src: Tensor) =
        ToroError.wrap (fun () ->
            use _scope = torch.no_grad ()
            inner.copy_ (src.Inner) |> ignore)

    // --- Misc ---

    member _.clone() =
        ToroError.wrap (fun () -> Tensor(inner.clone ()))

    // --- Scalar extraction ---

    member _.toFloat32Scalar() =
        ToroError.wrap (fun () -> inner.ToSingle())

    member _.toFloat64Scalar() =
        ToroError.wrap (fun () -> inner.ToDouble())

    member _.toInt32Scalar() =
        ToroError.wrap (fun () -> inner.ToInt32())

    member _.toInt64Scalar() =
        ToroError.wrap (fun () -> inner.ToInt64())

    // --- Operators (throw on error) ---

    static member (+)(a: Tensor, b: Tensor) = Tensor(a.Inner.add (b.Inner))

    static member (-)(a: Tensor, b: Tensor) = Tensor(a.Inner.sub (b.Inner))

    static member (*)(a: Tensor, b: Tensor) = Tensor(a.Inner.mul (b.Inner))

    static member (/)(a: Tensor, b: Tensor) = Tensor(a.Inner.div (b.Inner))

    static member (~-)(t: Tensor) = Tensor(t.Inner.neg ())

    static member (+)(t: Tensor, s: float) =
        Tensor(t.Inner.add (toScalar s: Scalar))

    static member (+)(s: float, t: Tensor) =
        Tensor(t.Inner.add (toScalar s: Scalar))

    static member (*)(t: Tensor, s: float) =
        Tensor(t.Inner.mul (toScalar s: Scalar))

    static member (*)(s: float, t: Tensor) =
        Tensor(t.Inner.mul (toScalar s: Scalar))

    static member (/)(t: Tensor, s: float) =
        Tensor(t.Inner.div (toScalar s: Scalar))

    // --- Disposal ---

    member _.Dispose() = inner.Dispose()

    interface System.IDisposable with
        member this.Dispose() = this.Dispose()

    // --- Display ---

    override _.ToString() =
        let shape = inner.shape |> Shape.ofInt64Array

        let dtype = DType.ofTorch inner.dtype
        $"Tensor[{shape}, {dtype}]"

module Toro =
    let noGrad (f: unit -> 'a) : 'a =
        use _scope = torch.no_grad ()
        f ()
