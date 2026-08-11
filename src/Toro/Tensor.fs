namespace Toro

open TorchSharp

[<AutoOpen>]
module internal ScalarHelper =
    let toScalar (v: float) : Scalar = Scalar.op_Implicit v

/// SRTP witness for Tensor.ofArray dispatch.
[<System.ComponentModel.EditorBrowsable(System.ComponentModel.EditorBrowsableState.Never)>]
type TensorOfArray = TensorOfArray
    with

        static member ($)(TensorOfArray, d: float32[]) =
            fun dev -> torch.tensor (d, device = dev)

        static member ($)(TensorOfArray, d: float[]) =
            fun dev -> torch.tensor (d, device = dev)

        static member ($)(TensorOfArray, d: int32[]) =
            fun dev -> torch.tensor (d, device = dev)

        static member ($)(TensorOfArray, d: int64[]) =
            fun dev -> torch.tensor (d, device = dev)

        static member ($)(TensorOfArray, d: float32[,]) =
            fun dev -> torch.tensor (d, device = dev)

        static member ($)(TensorOfArray, d: float[,]) =
            fun dev -> torch.tensor (d, device = dev)

        static member ($)(TensorOfArray, d: int32[,]) =
            fun dev -> torch.tensor (d, device = dev)

        static member ($)(TensorOfArray, d: int64[,]) =
            fun dev -> torch.tensor (d, device = dev)

        static member ($)(TensorOfArray, d: float32[,,]) =
            fun dev -> torch.tensor (d, device = dev)

        static member ($)(TensorOfArray, d: float[,,]) =
            fun dev -> torch.tensor (d, device = dev)

        static member ($)(TensorOfArray, d: int32[,,]) =
            fun dev -> torch.tensor (d, device = dev)

        static member ($)(TensorOfArray, d: int64[,,]) =
            fun dev -> torch.tensor (d, device = dev)

        static member ($)(TensorOfArray, d: float32[,,,]) =
            fun dev -> torch.tensor (d, device = dev)

        static member ($)(TensorOfArray, d: float[,,,]) =
            fun dev -> torch.tensor (d, device = dev)

        static member ($)(TensorOfArray, d: int32[,,,]) =
            fun dev -> torch.tensor (d, device = dev)

        static member ($)(TensorOfArray, d: int64[,,,]) =
            fun dev -> torch.tensor (d, device = dev)

/// SRTP witness for Tensor.ofList dispatch.
[<System.ComponentModel.EditorBrowsable(System.ComponentModel.EditorBrowsableState.Never)>]
type TensorOfList = TensorOfList
    with

        static member ($)(TensorOfList, d: float32 list) =
            fun dev -> torch.tensor (List.toArray d, device = dev)

        static member ($)(TensorOfList, d: float list) =
            fun dev -> torch.tensor (List.toArray d, device = dev)

        static member ($)(TensorOfList, d: int32 list) =
            fun dev -> torch.tensor (List.toArray d, device = dev)

        static member ($)(TensorOfList, d: int64 list) =
            fun dev -> torch.tensor (List.toArray d, device = dev)

        static member ($)(TensorOfList, d: float32 list list) =
            fun dev -> torch.tensor (array2D d, device = dev)

        static member ($)(TensorOfList, d: float list list) =
            fun dev -> torch.tensor (array2D d, device = dev)

        static member ($)(TensorOfList, d: int32 list list) =
            fun dev -> torch.tensor (array2D d, device = dev)

        static member ($)(TensorOfList, d: int64 list list) =
            fun dev -> torch.tensor (array2D d, device = dev)

/// Interpolation mode for resize operations.
type InterpolateMode =
    | Nearest
    | Bilinear
    | Bicubic

/// Wrapper around TorchSharp tensor. Most methods return Result&lt;'T, ToroError&gt;;
/// arithmetic operators throw on failure for ergonomic use in expressions.
type Tensor internal (inner: torch.Tensor) =

    /// Underlying TorchSharp tensor.
    member _.Inner = inner

    /// Shape of the tensor.
    member _.Shape = inner.shape |> Shape.ofInt64Array

    /// Number of dimensions.
    member _.Rank = int inner.ndim

    /// Data type of the elements.
    member _.DType = DType.ofTorch inner.dtype

    /// Device where the tensor resides.
    member _.Device = Device.ofTorch inner.device

    /// Total number of elements.
    member _.ElemCount = inner.NumberOfElements

    /// True if the tensor is contiguous in memory.
    member _.IsContiguous = inner.is_contiguous ()

    // --- Factory methods ---

    /// Create a tensor of zeros with the given shape.
    static member zeros(shape: int list, dtype: DType, device: Device) =
        ToroError.wrap (fun () ->
            let t =
                torch.zeros (Shape.toInt64Array shape, dtype = DType.toTorch dtype, device = Device.toTorch device)

            Tensor t)

    /// Create a tensor of ones with the given shape.
    static member ones(shape: int list, dtype: DType, device: Device) =
        ToroError.wrap (fun () ->
            let t =
                torch.ones (Shape.toInt64Array shape, dtype = DType.toTorch dtype, device = Device.toTorch device)

            Tensor t)

    /// Create a tensor filled with a scalar value.
    static member full(shape: int list, value: float, dtype: DType, device: Device) =
        ToroError.wrap (fun () ->
            let t =
                torch.full (
                    Shape.toInt64Array shape,
                    toScalar value,
                    dtype = DType.toTorch dtype,
                    device = Device.toTorch device
                )

            Tensor t)

    /// Create a tensor with uniform random values in [0, 1).
    static member rand(shape: int list, dtype: DType, device: Device) =
        ToroError.wrap (fun () ->
            let t =
                torch.rand (Shape.toInt64Array shape, dtype = DType.toTorch dtype, device = Device.toTorch device)

            Tensor t)

    /// Create a tensor with standard-normal random values.
    static member randn(shape: int list, dtype: DType, device: Device) =
        ToroError.wrap (fun () ->
            let t =
                torch.randn (Shape.toInt64Array shape, dtype = DType.toTorch dtype, device = Device.toTorch device)

            Tensor t)

    /// Create a 1-D tensor with values [0, stop).
    static member arange(stop: float, dtype: DType, device: Device) =
        ToroError.wrap (fun () ->
            let t =
                torch.arange (toScalar stop, dtype = DType.toTorch dtype, device = Device.toTorch device)

            Tensor t)

    /// Create a 1-D tensor with values [start, stop).
    static member arange(start: float, stop: float, dtype: DType, device: Device) =
        ToroError.wrap (fun () ->
            let t =
                torch.arange (toScalar start, toScalar stop, dtype = DType.toTorch dtype, device = Device.toTorch device)

            Tensor t)

    /// Create a tensor from an F# array (1-D through 4-D).
    static member inline ofArray(data, device: Device) =
        Tensor.ofTorchTensor ((TensorOfArray $ data) (Device.toTorch device))

    /// Create a tensor from an F# list (1-D or 2-D).
    static member inline ofList(data, device: Device) =
        Tensor.ofTorchTensor ((TensorOfList $ data) (Device.toTorch device))

    /// Concatenate tensors along a dimension.
    static member cat(tensors: Tensor list, dim: int) =
        ToroError.wrap (fun () ->
            let ts = tensors |> List.toArray |> Array.map _.Inner

            Tensor(torch.cat (ts, int64 dim)))

    /// Stack tensors along a new dimension.
    static member stack(tensors: Tensor list, dim: int) =
        ToroError.wrap (fun () ->
            let ts = tensors |> List.toArray |> Array.map _.Inner

            Tensor(torch.stack (ts, int64 dim)))

    /// Wrap a raw TorchSharp tensor.
    static member ofTorchTensor(t: torch.Tensor) = ToroError.wrap (fun () -> Tensor t)

    // --- Dimension query ---

    /// Return the size of dimension d.
    member t.dim(d: int) =
        ToroError.wrap (fun () -> inner.size d |> int)

    // --- Arithmetic (tensor-tensor) ---

    /// Elementwise addition.
    member _.add(other: Tensor) =
        ToroError.wrap (fun () -> Tensor(inner.add other.Inner))

    /// Elementwise subtraction.
    member _.sub(other: Tensor) =
        ToroError.wrap (fun () -> Tensor(inner.sub other.Inner))

    /// Elementwise multiplication.
    member _.mul(other: Tensor) =
        ToroError.wrap (fun () -> Tensor(inner.mul other.Inner))

    /// Elementwise division.
    member _.div(other: Tensor) =
        ToroError.wrap (fun () -> Tensor(inner.div other.Inner))

    // --- Arithmetic (scalar) ---

    /// Add a scalar to each element.
    member _.addScalar(s: float) =
        ToroError.wrap (fun () -> Tensor(inner.add (toScalar s: Scalar)))

    /// Multiply each element by a scalar.
    member _.mulScalar(s: float) =
        ToroError.wrap (fun () -> Tensor(inner.mul (toScalar s: Scalar)))

    /// Subtract a scalar from each element.
    member _.subScalar(s: float) =
        ToroError.wrap (fun () -> Tensor(inner.sub (toScalar s: Scalar)))

    /// Divide each element by a scalar.
    member _.divScalar(s: float) =
        ToroError.wrap (fun () -> Tensor(inner.div (toScalar s: Scalar)))

    // --- Matrix ops ---

    /// Matrix multiplication.
    member _.matmul(other: Tensor) =
        ToroError.wrap (fun () -> Tensor(inner.matmul other.Inner))

    /// Transpose a 2-D tensor.
    member _.t() =
        ToroError.wrap (fun () -> Tensor(inner.t ()))

    /// Swap two dimensions.
    member _.transpose(dim0: int, dim1: int) =
        ToroError.wrap (fun () -> Tensor(inner.transpose (int64 dim0, int64 dim1)))

    // --- Shape manipulation ---

    /// Reshape to the given shape.
    member _.reshape(shape: int list) =
        ToroError.wrap (fun () -> Tensor(inner.reshape (Shape.toInt64Array shape)))

    /// View with the given shape (must be contiguous).
    member _.view(shape: int list) =
        ToroError.wrap (fun () -> Tensor(inner.view (Shape.toInt64Array shape)))

    /// Flatten dimensions [startDim, endDim].
    member _.flatten(startDim: int, endDim: int) =
        ToroError.wrap (fun () -> Tensor(inner.flatten (int64 startDim, int64 endDim)))

    /// Flatten all dimensions to 1-D.
    member _.flattenAll() =
        ToroError.wrap (fun () -> Tensor(inner.flatten (0L, -1L)))

    /// Remove a size-1 dimension.
    member _.squeeze(dim: int) =
        ToroError.wrap (fun () -> Tensor(inner.squeeze (int64 dim)))

    /// Insert a size-1 dimension.
    member _.unsqueeze(dim: int) =
        ToroError.wrap (fun () -> Tensor(inner.unsqueeze (int64 dim)))

    /// Return a contiguous copy.
    member _.contiguous() =
        ToroError.wrap (fun () -> Tensor(inner.contiguous ()))

    /// Reorder dimensions.
    member _.permute(dims: int list) =
        ToroError.wrap (fun () -> Tensor(inner.permute (dims |> List.map int64 |> List.toArray)))

    /// Reverse elements along the given dimensions.
    member _.flip(dims: int list) =
        ToroError.wrap (fun () -> Tensor(inner.flip (dims |> List.map int64 |> List.toArray)))

    /// Broadcast to the given shape.
    member _.expand(shape: int list) =
        ToroError.wrap (fun () -> Tensor(inner.expand (Shape.toInt64Array shape)))

    /// Repeat elements along a dimension.
    member _.repeatInterleave(repeats: int, dim: int) =
        ToroError.wrap (fun () -> Tensor(inner.repeat_interleave (int64 repeats, int64 dim)))

    /// Pad with a constant value.
    member _.pad(padding: int list, value: float) =
        ToroError.wrap (fun () ->
            let p = padding |> List.map int64 |> List.toArray
            Tensor(torch.nn.functional.pad (inner, p, value = value)))

    /// Lower-triangular part.
    member _.tril(?diagonal: int) =
        let d = defaultArg diagonal 0
        ToroError.wrap (fun () -> Tensor(inner.tril (int64 d)))

    /// Upper-triangular part.
    member _.triu(?diagonal: int) =
        let d = defaultArg diagonal 0
        ToroError.wrap (fun () -> Tensor(inner.triu (int64 d)))

    /// Broadcast to the given shape (left-aligned).
    member _.broadcastLeft(shape: int list) =
        ToroError.wrap (fun () -> Tensor(inner.broadcast_to (Shape.toInt64Array shape)))

    // --- Reduction ---

    /// Sum of all elements.
    member _.sumAll() =
        ToroError.wrap (fun () -> Tensor(inner.sum ()))

    /// Sum along a dimension.
    member _.sum(dim: int, ?keepDim: bool) =
        let kd = defaultArg keepDim false

        ToroError.wrap (fun () -> Tensor(inner.sum ([| int64 dim |], keepdim = kd)))

    /// Mean of all elements.
    member _.meanAll() =
        ToroError.wrap (fun () -> Tensor(inner.mean ()))

    /// Mean along a dimension.
    member _.mean(dim: int, ?keepDim: bool) =
        let kd = defaultArg keepDim false

        ToroError.wrap (fun () -> Tensor(inner.mean ([| int64 dim |], keepdim = kd)))

    /// Index of the maximum along a dimension.
    member _.argmax(dim: int, ?keepDim: bool) =
        let kd = defaultArg keepDim false
        ToroError.wrap (fun () -> Tensor(inner.argmax (int64 dim, kd)))

    /// Index of the minimum along a dimension.
    member _.argmin(dim: int, ?keepDim: bool) =
        let kd = defaultArg keepDim false
        ToroError.wrap (fun () -> Tensor(inner.argmin (int64 dim, kd)))

    /// Maximum values and indices along a dimension.
    member _.max(dim: int, ?keepDim: bool) =
        let kd = defaultArg keepDim false

        ToroError.wrap (fun () ->
            let struct (values, indices) = inner.max (int64 dim, kd)
            Tensor values, Tensor indices)

    /// Minimum values and indices along a dimension.
    member _.min(dim: int, ?keepDim: bool) =
        let kd = defaultArg keepDim false

        ToroError.wrap (fun () ->
            let struct (values, indices) = inner.min (int64 dim, kd)
            Tensor values, Tensor indices)

    /// Select elements from x or y by condition.
    static member where(condition: Tensor, x: Tensor, y: Tensor) =
        ToroError.wrap (fun () -> Tensor(torch.where (condition.Inner, x.Inner, y.Inner)))

    // --- Unary ops ---

    /// $-x$
    member _.neg() =
        ToroError.wrap (fun () -> Tensor(inner.neg ()))

    /// $\lvert x \rvert$
    member _.abs() =
        ToroError.wrap (fun () -> Tensor(inner.abs ()))

    /// $\sqrt{x}$
    member _.sqrt() =
        ToroError.wrap (fun () -> Tensor(inner.sqrt ()))

    /// $x^2$
    member _.sqr() =
        ToroError.wrap (fun () -> Tensor(inner.square ()))

    /// $x^n$
    member _.pow(exponent: float) =
        ToroError.wrap (fun () -> Tensor(inner.pow (toScalar exponent)))

    /// $e^x$
    member _.exp() =
        ToroError.wrap (fun () -> Tensor(inner.exp ()))

    /// $\ln x$
    member _.log() =
        ToroError.wrap (fun () -> Tensor(inner.log ()))

    /// $\max(0, x)$
    member _.relu() =
        ToroError.wrap (fun () -> Tensor(inner.relu ()))

    /// $\text{GELU}(x) = x \cdot \Phi(x)$
    member _.gelu() =
        ToroError.wrap (fun () -> Tensor(torch.nn.functional.gelu inner))

    /// $\text{SiLU}(x) = x \cdot \sigma(x)$
    member _.silu() =
        ToroError.wrap (fun () -> Tensor(torch.nn.functional.silu inner))

    /// $\tanh(x)$
    member _.tanh() =
        ToroError.wrap (fun () -> Tensor(inner.tanh ()))

    /// $\sigma(x) = 1 / (1+e^{-x})$
    member _.sigmoid() =
        ToroError.wrap (fun () -> Tensor(inner.sigmoid ()))

    /// $\max(\alpha x, x)$
    member _.leakyRelu(negativeSlope: float) =
        ToroError.wrap (fun () -> Tensor(torch.nn.functional.leaky_relu (inner, negativeSlope)))

    /// $\text{ELU}(x) = \max(0,x) + \min(0, \alpha(e^x - 1))$
    member _.elu(alpha: float) =
        ToroError.wrap (fun () -> Tensor(torch.nn.functional.elu (inner, alpha)))

    /// $x \cdot \tanh(\text{softplus}(x))$
    member _.mish() =
        ToroError.wrap (fun () -> Tensor(torch.nn.functional.mish inner))

    /// Randomly zero elements with probability p.
    member _.dropout(p: float, train: bool) =
        ToroError.wrap (fun () -> Tensor(torch.nn.functional.dropout (inner, p, train)))

    /// $\text{softmax}(x_i) = e^{x_i} / \sum_j e^{x_j}$
    member _.softmax(dim: int) =
        ToroError.wrap (fun () -> Tensor(torch.nn.functional.softmax (inner, int64 dim)))

    /// $\log\text{softmax}(x_i) = x_i - \log\sum_j e^{x_j}$
    member _.logSoftmax(dim: int) =
        ToroError.wrap (fun () -> Tensor(torch.nn.functional.log_softmax (inner, int64 dim)))

    /// Clamp elements to [min, max].
    member _.clamp(min: float, max: float) =
        ToroError.wrap (fun () -> Tensor(inner.clamp (toScalar min, toScalar max)))

    /// $ax + b$
    member _.affine(mul: float, add: float) =
        ToroError.wrap (fun () ->
            let t = inner.mul (toScalar mul: Scalar)
            Tensor(t.add (toScalar add: Scalar)))

    // --- Indexing ---

    /// Select slices along a dimension by index.
    member _.indexSelect(dim: int, index: Tensor) =
        ToroError.wrap (fun () -> Tensor(inner.index_select (int64 dim, index.Inner)))

    /// Gather elements along a dimension by index.
    member _.gather(dim: int, index: Tensor) =
        ToroError.wrap (fun () -> Tensor(inner.gather (int64 dim, index.Inner)))

    /// Narrow a dimension to [start, start+length).
    member _.narrow(dim: int, start: int64, length: int64) =
        ToroError.wrap (fun () -> Tensor(inner.narrow (int64 dim, start, length)))

    /// Scatter-add src into self along dim using index.
    member _.scatterAdd(dim: int, index: Tensor, src: Tensor) =
        ToroError.wrap (fun () -> Tensor(inner.scatter_add (int64 dim, index.Inner, src.Inner)))

    /// Add src into self along dim at positions given by index.
    member _.indexAdd(dim: int, index: Tensor, src: Tensor) =
        ToroError.wrap (fun () -> Tensor(inner.index_add (int64 dim, index.Inner, src.Inner, toScalar 1.0)))

    /// Split into chunks along a dimension.
    member _.chunk(chunks: int, dim: int) =
        ToroError.wrap (fun () ->
            inner.chunk (int64 chunks, int64 dim)
            |> Array.toList
            |> List.map Tensor)

    // --- Type / Device conversion ---

    /// Move to a device.
    member _.toDevice(device: Device) =
        ToroError.wrap (fun () -> Tensor(inner.``to`` (Device.toTorch device)))

    /// Cast to a data type.
    member _.toDType(dtype: DType) =
        ToroError.wrap (fun () -> Tensor(inner.``to`` (DType.toTorch dtype)))

    // --- Autograd ---

    /// True if gradient tracking is enabled.
    member _.RequiresGrad = inner.requires_grad

    /// Enable or disable gradient tracking.
    member _.requiresGrad(?requiresGrad: bool) =
        let rg = defaultArg requiresGrad true

        ToroError.wrap (fun () -> Tensor(inner.requires_grad_ rg))

    /// Compute gradients by backpropagation.
    member _.backward() =
        ToroError.wrap (fun () -> inner.backward ())

    /// Return the accumulated gradient.
    member _.grad() =
        ToroError.wrap (fun () ->
            if isNull inner.grad then
                Tensor(torch.zeros_like inner)
            else
                Tensor inner.grad)

    /// Detach from the computation graph.
    member _.detach() =
        ToroError.wrap (fun () -> Tensor(inner.detach ()))

    /// Zero the accumulated gradient.
    member _.zeroGrad() =
        if not (isNull inner.grad) then
            inner.grad.zero_ () |> ignore

    /// Copy data from src without gradient tracking.
    member _.copyInPlace(src: Tensor) =
        ToroError.wrap (fun () ->
            use _scope = torch.no_grad ()
            inner.copy_ src.Inner |> ignore)

    // --- Convolution ---

    /// Apply 1-D convolution.
    member _.conv1d(weight: Tensor, ?bias: Tensor, ?stride: int, ?padding: int, ?dilation: int, ?groups: int) =
        ToroError.wrap (fun () ->
            let s = int64 (defaultArg stride 1)
            let p = int64 (defaultArg padding 0)
            let d = int64 (defaultArg dilation 1)
            let g = int64 (defaultArg groups 1)

            let b = bias |> Option.map _.Inner |> Option.defaultValue null

            Tensor(torch.nn.functional.conv1d (inner, weight.Inner, b, s, p, d, g)))

    /// Apply 2-D convolution.
    member _.conv2d(weight: Tensor, ?bias: Tensor, ?stride: int, ?padding: int, ?dilation: int, ?groups: int) =
        ToroError.wrap (fun () ->
            let s = int64 (defaultArg stride 1)
            let p = int64 (defaultArg padding 0)
            let d = int64 (defaultArg dilation 1)
            let g = int64 (defaultArg groups 1)

            let b = bias |> Option.map _.Inner |> Option.defaultValue null

            Tensor(torch.nn.functional.conv2d (inner, weight.Inner, b, [| s; s |], [| p; p |], [| d; d |], g)))

    // --- Normalization ---

    /// Apply batch normalization.
    member _.batchNorm
        (
            weight: Tensor option,
            bias: Tensor option,
            runningMean: Tensor option,
            runningVar: Tensor option,
            train: bool,
            momentum: float,
            eps: float
        ) =
        ToroError.wrap (fun () ->
            let w = weight |> Option.map _.Inner |> Option.defaultValue null

            let b = bias |> Option.map _.Inner |> Option.defaultValue null

            let rm =
                runningMean
                |> Option.map _.Inner
                |> Option.defaultValue null

            let rv = runningVar |> Option.map _.Inner |> Option.defaultValue null

            Tensor(torch.nn.functional.batch_norm (inner, rm, rv, w, b, train, momentum, eps)))

    /// Apply group normalization.
    member _.groupNorm(numGroups: int, ?weight: Tensor, ?bias: Tensor, ?eps: float) =
        ToroError.wrap (fun () ->
            let e = defaultArg eps 1e-5

            let w = weight |> Option.map _.Inner |> Option.defaultValue null

            let b = bias |> Option.map _.Inner |> Option.defaultValue null

            Tensor(torch.nn.functional.group_norm (inner, int64 numGroups, w, b, e)))

    // --- Pooling ---

    /// Apply 1-D max pooling.
    member _.maxPool1d(kernelSize: int, ?stride: int, ?padding: int) =
        ToroError.wrap (fun () ->
            let s = int64 (defaultArg stride kernelSize)
            let p = int64 (defaultArg padding 0)

            Tensor(torch.nn.functional.max_pool1d (inner, int64 kernelSize, stride = s, padding = p)))

    /// Apply 2-D max pooling.
    member _.maxPool2d(kernelSize: int, ?stride: int, ?padding: int) =
        ToroError.wrap (fun () ->
            let s = int64 (defaultArg stride kernelSize)
            let p = int64 (defaultArg padding 0)

            Tensor(torch.nn.functional.max_pool2d (inner, int64 kernelSize, stride = s, padding = p)))

    /// Apply 2-D average pooling.
    member _.avgPool2d(kernelSize: int, ?stride: int, ?padding: int) =
        ToroError.wrap (fun () ->
            let s = int64 (defaultArg stride kernelSize)
            let p = int64 (defaultArg padding 0)

            Tensor(torch.nn.functional.avg_pool2d (inner, int64 kernelSize, stride = s, padding = p)))

    /// Resize a 4-D tensor $[B, C, H, W]$ to the target spatial size.
    member _.interpolate(size: int list, mode: InterpolateMode) =
        ToroError.wrap (fun () ->
            let sz = size |> List.map int64 |> List.toArray

            let m =
                match mode with
                | Nearest -> torch.InterpolationMode.Nearest
                | Bilinear -> torch.InterpolationMode.Bilinear
                | Bicubic -> torch.InterpolationMode.Bicubic

            Tensor(torch.nn.functional.interpolate (inner, sz, mode = m)))

    // --- Attention ---

    /// $\text{Attention}(Q,K,V) = \text{softmax}(QK^\top / \sqrt{d_k})\,V$
    member _.scaledDotProductAttention(key: Tensor, value: Tensor, ?attnMask: Tensor, ?dropoutP: float, ?isCausal: bool) =
        ToroError.wrap (fun () ->
            let dp = defaultArg dropoutP 0.0
            let causal = defaultArg isCausal false

            let mask = attnMask |> Option.map _.Inner |> Option.defaultValue null

            Tensor(
                torch.nn.functional.scaled_dot_product_attention (
                    inner,
                    key.Inner,
                    value.Inner,
                    attn_mask = mask,
                    p = dp,
                    is_casual = causal
                )
            ))

    /// Fill positions where mask is true with a value.
    member _.maskedFill(mask: Tensor, value: float) =
        ToroError.wrap (fun () -> Tensor(inner.masked_fill (mask.Inner, toScalar value)))

    /// Create a causal attention mask.
    static member causalMask(seqLen: int, dtype: DType, device: Device) =
        ToroError.wrap (fun () ->
            use _scope = torch.NewDisposeScope()

            let ones =
                torch.ones (int64 seqLen, int64 seqLen, dtype = torch.bool, device = Device.toTorch device)

            let mask = ones.triu (1L)

            let filled =
                torch.zeros (int64 seqLen, int64 seqLen, dtype = DType.toTorch dtype, device = Device.toTorch device)

            let r = Tensor(filled.masked_fill (mask, toScalar System.Double.NegativeInfinity))
            r.Inner.MoveToOuterDisposeScope() |> ignore
            r)

    // --- Encoding ---

    /// One-hot encode along numClasses.
    member _.oneHot(numClasses: int) =
        ToroError.wrap (fun () -> Tensor(torch.nn.functional.one_hot(inner, int64 numClasses).``to`` torch.float32))

    // --- Misc ---

    /// Return a deep copy.
    member _.clone() =
        ToroError.wrap (fun () -> Tensor(inner.clone ()))

    // --- Persistence ---

    /// Save to a file.
    member _.save(path: string) =
        ToroError.wrap (fun () -> inner.save path)

    /// Load a tensor from a file.
    static member load(path: string) =
        ToroError.wrap (fun () -> Tensor(torch.Tensor.load path))

    // --- Scalar extraction ---

    /// Extract a float32 scalar.
    member _.toFloat32Scalar() =
        ToroError.wrap (fun () -> inner.ToSingle())

    /// Extract a float64 scalar.
    member _.toFloat64Scalar() =
        ToroError.wrap (fun () -> inner.ToDouble())

    /// Extract an int32 scalar.
    member _.toInt32Scalar() =
        ToroError.wrap (fun () -> inner.ToInt32())

    /// Extract an int64 scalar.
    member _.toInt64Scalar() =
        ToroError.wrap (fun () -> inner.ToInt64())

    /// Extract as float64 (throw on error).
    member _.item() : float = inner.ToDouble()

    /// Extract as float32 (throw on error).
    member _.itemF32() : float32 = inner.ToSingle()

    /// Extract as int64 (throw on error).
    member _.itemI64() : int64 = inner.ToInt64()

    /// Extract as int32 (throw on error).
    member _.itemI32() : int = inner.ToInt32()

    // --- Indexers (throw on error) ---

    /// Get element at index i.
    member _.Item
        with get (i: int): Tensor = Tensor(inner.index (torch.TensorIndex.Single(int64 i)))

    /// Index by a tensor.
    member _.Item
        with get (idx: Tensor): Tensor = Tensor(inner.index (torch.TensorIndex.Tensor(idx.Inner)))

    /// Slice the first dimension.
    member _.GetSlice(startIdx: int option, endIdx: int option) : Tensor =
        let s = startIdx |> Option.map int64 |> Option.toNullable

        let e =
            match endIdx with
            | None -> System.Nullable()
            | Some -1 -> System.Nullable()
            | Some e -> System.Nullable(int64 (e + 1))

        Tensor(inner.index (torch.TensorIndex.Slice(s, e)))

    /// Index by a list of TIdx specifiers.
    member _.at(indices: TIdx list) : Tensor =
        let toTorchIndex =
            function
            | TIdx.I i -> torch.TensorIndex.Single(int64 i)
            | TIdx.S(s, e) -> torch.TensorIndex.Slice(System.Nullable(int64 s), System.Nullable(int64 e))
            | TIdx.Sf s -> torch.TensorIndex.Slice(System.Nullable(int64 s), System.Nullable())
            | TIdx.St e -> torch.TensorIndex.Slice(System.Nullable(), System.Nullable(int64 e))
            | TIdx.A -> torch.TensorIndex.Slice()
            | TIdx.T t -> torch.TensorIndex.Tensor(t.Inner)
            | TIdx.E -> torch.TensorIndex.Ellipsis
            | TIdx.N -> torch.TensorIndex.None

        let tIndices = indices |> List.map toTorchIndex |> List.toArray
        Tensor(inner.index tIndices)

    // --- Operators (throw on error) ---

    /// $a + b$
    static member (+)(a: Tensor, b: Tensor) = Tensor(a.Inner.add b.Inner)

    /// $a - b$
    static member (-)(a: Tensor, b: Tensor) = Tensor(a.Inner.sub b.Inner)

    /// $a \times b$
    static member (*)(a: Tensor, b: Tensor) = Tensor(a.Inner.mul b.Inner)

    /// $a / b$
    static member (/)(a: Tensor, b: Tensor) = Tensor(a.Inner.div b.Inner)

    /// $-t$
    static member (~-)(t: Tensor) = Tensor(t.Inner.neg ())

    /// $t + s$
    static member (+)(t: Tensor, s: float) =
        Tensor(t.Inner.add (toScalar s: Scalar))

    /// $s + t$
    static member (+)(s: float, t: Tensor) =
        Tensor(t.Inner.add (toScalar s: Scalar))

    /// $t \times s$
    static member (*)(t: Tensor, s: float) =
        Tensor(t.Inner.mul (toScalar s: Scalar))

    /// $s \times t$
    static member (*)(s: float, t: Tensor) =
        Tensor(t.Inner.mul (toScalar s: Scalar))

    /// $t - s$
    static member (-)(t: Tensor, s: float) =
        Tensor(t.Inner.sub (toScalar s: Scalar))

    /// $s - t$
    static member (-)(s: float, t: Tensor) =
        Tensor(t.Inner.neg().add (toScalar s: Scalar))

    /// $t / s$
    static member (/)(t: Tensor, s: float) =
        Tensor(t.Inner.div (toScalar s: Scalar))

    // --- Comparison (throw on error) ---

    /// Elementwise equal.
    member _.eq(other: Tensor) = Tensor(inner.eq other.Inner)

    /// Elementwise not-equal.
    member _.ne(other: Tensor) = Tensor(inner.ne other.Inner)

    /// Elementwise greater-than.
    member _.gt(other: Tensor) = Tensor(inner.gt other.Inner)

    /// Elementwise less-than.
    member _.lt(other: Tensor) = Tensor(inner.lt other.Inner)

    /// Elementwise greater-or-equal.
    member _.ge(other: Tensor) = Tensor(inner.ge other.Inner)

    /// Elementwise less-or-equal.
    member _.le(other: Tensor) = Tensor(inner.le other.Inner)

    /// Elementwise equal to a scalar.
    member _.eqScalar(s: float) = Tensor(inner.eq (toScalar s))

    /// Elementwise not-equal to a scalar.
    member _.neScalar(s: float) = Tensor(inner.ne (toScalar s))

    /// Elementwise greater-than a scalar.
    member _.gtScalar(s: float) = Tensor(inner.gt (toScalar s))

    /// Elementwise less-than a scalar.
    member _.ltScalar(s: float) = Tensor(inner.lt (toScalar s))

    /// Elementwise greater-or-equal to a scalar.
    member _.geScalar(s: float) = Tensor(inner.ge (toScalar s))

    /// Elementwise less-or-equal to a scalar.
    member _.leScalar(s: float) = Tensor(inner.le (toScalar s))

    /// Elementwise $a = b$.
    static member (.=.)(a: Tensor, b: Tensor) = Tensor(a.Inner.eq b.Inner)

    /// Elementwise $a \neq b$.
    static member (.<>.)(a: Tensor, b: Tensor) = Tensor(a.Inner.ne b.Inner)

    /// Elementwise $a > b$.
    static member (.>.)(a: Tensor, b: Tensor) = Tensor(a.Inner.gt b.Inner)

    /// Elementwise $a < b$.
    static member (.<.)(a: Tensor, b: Tensor) = Tensor(a.Inner.lt b.Inner)

    /// Elementwise $a \geq b$.
    static member (.>=.)(a: Tensor, b: Tensor) = Tensor(a.Inner.ge b.Inner)

    /// Elementwise $a \leq b$.
    static member (.<=.)(a: Tensor, b: Tensor) = Tensor(a.Inner.le b.Inner)

    // --- Disposal ---

    member _.Dispose() = inner.Dispose()

    interface System.IDisposable with
        member this.Dispose() = this.Dispose()

    // --- Display ---

    override _.ToString() =
        let shape = inner.shape |> Shape.ofInt64Array

        let dtype = DType.ofTorch inner.dtype
        $"Tensor[{shape}, {dtype}]"

/// Index specifier for <c>Tensor.at</c>. I=single, S=slice, A=all, T=tensor, E=ellipsis, N=newaxis.
and TIdx =
    | I of int
    | S of start: int * stop: int
    | Sf of start: int
    | St of stop: int
    | A
    | T of Tensor
    | E
    | N

module Tensor =
    /// Move a tensor out of the current dispose scope so it
    /// survives when that scope exits.
    let keep (t: Tensor) : Tensor =
        t.Inner.MoveToOuterDisposeScope() |> ignore
        t

module Toro =
    /// Run f with gradient tracking disabled.
    let noGrad (f: unit -> 'a) : 'a =
        use _scope = torch.no_grad ()
        f ()

/// <c>result { }</c> with automatic disposal of intermediate TorchSharp tensors.
/// All <c>torch.Tensor</c> objects created inside the block are disposed when
/// the scope exits. Tensors in the return value (including inside tuples)
/// are automatically kept alive past the scope.
type ScopedResultBuilder() =
    static let flags =
        System.Reflection.BindingFlags.Public
        ||| System.Reflection.BindingFlags.NonPublic

    static let rec keepTensors (scope: DisposeScope) (v: obj) =
        if isNull v then
            ()
        else
            match v with
            | :? Tensor as t ->
                if scope.Contains(t.Inner) then
                    scope.MoveToOuter(t.Inner) |> ignore
            | :? System.Collections.IEnumerable as xs ->
                for item in xs do
                    keepTensors scope item
            | _ ->
                let ty = v.GetType()

                if Microsoft.FSharp.Reflection.FSharpType.IsTuple ty then
                    for field in Microsoft.FSharp.Reflection.FSharpValue.GetTupleFields v do
                        keepTensors scope field
                elif Microsoft.FSharp.Reflection.FSharpType.IsRecord(ty, flags) then
                    for field in Microsoft.FSharp.Reflection.FSharpValue.GetRecordFields(v, flags) do
                        keepTensors scope field
                elif Microsoft.FSharp.Reflection.FSharpType.IsUnion(ty, flags) then
                    let _, fields = Microsoft.FSharp.Reflection.FSharpValue.GetUnionFields(v, ty, flags)

                    for field in fields do
                        keepTensors scope field

    member _.Return x = Ok x
    member _.ReturnFrom x = x

    member _.Bind(m, f) =
        match m with
        | Ok x -> f x
        | Error e -> Error e

    member _.Zero() = Ok()

    member _.Combine(m: Result<unit, 'e>, f: unit -> Result<'b, 'e>) =
        match m with
        | Ok() -> f ()
        | Error e -> Error e

    member _.Delay f = f

    member _.Run(f: unit -> Result<'a, ToroError>) : Result<'a, ToroError> =
        use scope = torch.NewDisposeScope()
        let r = f ()

        match r with
        | Ok v -> keepTensors scope (box v)
        | _ -> ()

        r

    member _.TryWith(body, handler) =
        try
            body ()
        with ex ->
            handler ex

    member _.TryFinally(body, finalizer) =
        try
            body ()
        finally
            finalizer ()

    member _.Using(resource: #System.IDisposable, body) =
        try
            body resource
        finally
            if not (isNull (box resource)) then
                resource.Dispose()

    member this.While(guard, body) =
        if not (guard ()) then
            this.Zero()
        else
            this.Bind(body (), (fun () -> this.While(guard, body)))

    member this.For(sequence: seq<'a>, body) =
        this.Using(sequence.GetEnumerator(), fun enum -> this.While(enum.MoveNext, this.Delay(fun () -> body enum.Current)))

[<AutoOpen>]
module ScopedCE =
    /// Computation expression that wraps <c>result { }</c> with a
    /// <c>torch.NewDisposeScope()</c>. Intermediate tensors are disposed
    /// automatically when the block completes.
    let scoped = ScopedResultBuilder()

    /// Start a TorchSharp dispose scope for use with <c>use!</c> inside
    /// <c>result { }</c>. Intermediate tensors created after this point
    /// are disposed when the binding goes out of scope.
    /// Unlike <c>scoped { }</c>, return values are NOT auto-kept.
    /// Use <c>Tensor.keep</c> to preserve tensors past the scope.
    let disposeScope () : Result<System.IDisposable, ToroError> = Ok(torch.NewDisposeScope())
