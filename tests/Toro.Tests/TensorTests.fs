module TensorTests

open Xunit
open FsUnit.Xunit
open Toro
open TorchSharp
open TestHelper

[<Fact>]
let ``zeros creates tensor with correct shape`` () =
    let t = torch.zeros ([| 2L; 3L |], dtype = torch.float32, device = torch.CPU)

    t.shape |> should equal [| 2L; 3L |]
    t.dtype |> should equal torch.float32
    t.device.``type`` |> should equal DeviceType.CPU

[<Fact>]
let ``ones creates tensor with correct shape`` () =
    let t = torch.ones ([| 4L; 5L |], dtype = torch.float64, device = torch.CPU)

    t.shape |> should equal [| 4L; 5L |]
    t.dtype |> should equal torch.float64

[<Fact>]
let ``randn creates tensor with correct shape`` () =
    let t = torch.randn ([| 3L; 4L |], dtype = torch.float32, device = torch.CPU)

    t.shape |> should equal [| 3L; 4L |]
    int t.ndim |> should equal 2
    t.NumberOfElements |> should equal 12L

[<Fact>]
let ``full creates tensor with given value`` () =
    let t =
        torch.full ([| 2L; 2L |], scalar 3.14, dtype = torch.float64, device = torch.CPU)

    t.shape |> should equal [| 2L; 2L |]
    t.dtype |> should equal torch.float64
    scalarF64 t |> should (equalWithin 1e-10) (3.14 * 4.0)

[<Fact>]
let ``matmul produces correct shape`` () =
    let a = torch.randn ([| 2L; 3L |], dtype = torch.float32, device = torch.CPU)
    let b = torch.randn ([| 3L; 4L |], dtype = torch.float32, device = torch.CPU)

    let c = a.matmul b
    c.shape |> should equal [| 2L; 4L |]

[<Fact>]
let ``add and sub are consistent`` () =
    let a = torch.ones ([| 2L; 2L |], dtype = torch.float32, device = torch.CPU)
    let b = torch.ones ([| 2L; 2L |], dtype = torch.float32, device = torch.CPU)

    let diff = (a.add b).sub b
    scalarF32 diff |> should equal 4.0f

[<Fact>]
let ``div divides tensors element-wise`` () =
    let a = torch.full ([| 3L |], scalar 6.0, dtype = torch.float32, device = torch.CPU)
    let b = torch.full ([| 3L |], scalar 2.0, dtype = torch.float32, device = torch.CPU)
    let c = a.div b
    scalarF32 c |> should equal 9.0f

[<Fact>]
let ``divScalar divides by scalar`` () =
    let t = torch.full ([| 3L |], scalar 6.0, dtype = torch.float32, device = torch.CPU)
    let s = t.div (scalar 3.0)
    scalarF32 s |> should equal 6.0f

[<Fact>]
let ``subScalar subtracts scalar value`` () =
    let t = torch.full ([| 3L |], scalar 5.0, dtype = torch.float32, device = torch.CPU)
    let s = t.sub (scalar 2.0)
    scalarF32 s |> should equal 9.0f

[<Fact>]
let ``pow computes element-wise power`` () =
    let t = torch.full ([| 3L |], scalar 2.0, dtype = torch.float64, device = torch.CPU)
    let p = t.pow (scalar 3.0)
    scalarF64 p |> should (equalWithin 1e-10) 24.0

[<Fact>]
let ``reshape changes shape`` () =
    let t = torch.randn ([| 2L; 3L |], dtype = torch.float32, device = torch.CPU)

    let r = t.reshape [| 6L |]
    r.shape |> should equal [| 6L |]
    int r.ndim |> should equal 1

[<Fact>]
let ``view reshapes tensor`` () =
    let t = torch.randn ([| 2L; 6L |], dtype = torch.float32, device = torch.CPU)
    let v = t.view [| 3L; 4L |]
    v.shape |> should equal [| 3L; 4L |]

[<Fact>]
let ``flatten collapses dimensions`` () =
    let t = torch.randn ([| 2L; 3L; 4L |], dtype = torch.float32, device = torch.CPU)
    let f = t.flatten (1L, 2L)
    f.shape |> should equal [| 2L; 12L |]

[<Fact>]
let ``flattenAll creates 1D tensor`` () =
    let t = torch.randn ([| 2L; 3L; 4L |], dtype = torch.float32, device = torch.CPU)
    let f = t.flatten (0L, -1L)
    f.shape |> should equal [| 24L |]

[<Fact>]
let ``transpose swaps dimensions`` () =
    let t = torch.randn ([| 2L; 3L |], dtype = torch.float32, device = torch.CPU)

    let tr = t.t ()
    tr.shape |> should equal [| 3L; 2L |]

[<Fact>]
let ``squeeze removes dim of size 1`` () =
    let t = torch.ones ([| 2L; 1L; 3L |], dtype = torch.float32, device = torch.CPU)
    let s = t.squeeze (1L)
    s.shape |> should equal [| 2L; 3L |]

[<Fact>]
let ``unsqueeze adds dim of size 1`` () =
    let t = torch.ones ([| 2L; 3L |], dtype = torch.float32, device = torch.CPU)
    let u = t.unsqueeze (1L)
    u.shape |> should equal [| 2L; 1L; 3L |]

[<Fact>]
let ``sum keepDim preserves rank`` () =
    let t = torch.ones ([| 2L; 3L; 4L |], dtype = torch.float32, device = torch.CPU)

    let s = t.sum ([| -1L |], keepdim = true)
    s.shape |> should equal [| 2L; 3L; 1L |]

[<Fact>]
let ``unary ops preserve shape`` () =
    let t = torch.ones ([| 2L; 3L |], dtype = torch.float32, device = torch.CPU)

    (t.neg ()).shape |> should equal [| 2L; 3L |]
    (t.abs ()).shape |> should equal [| 2L; 3L |]
    (t.sqrt ()).shape |> should equal [| 2L; 3L |]
    (t.exp ()).shape |> should equal [| 2L; 3L |]
    (t.log ()).shape |> should equal [| 2L; 3L |]

[<Fact>]
let ``neg negates values`` () =
    let t = torch.ones ([| 3L |], dtype = torch.float32, device = torch.CPU)
    let n = t.neg ()
    scalarF32 n |> should equal -3.0f

[<Fact>]
let ``clone creates independent copy`` () =
    let t = torch.ones ([| 3L |], dtype = torch.float32, device = torch.CPU)
    let c = t.clone ()
    c.shape |> should equal [| 3L |]
    scalarF32 c |> should equal 3.0f

[<Fact>]
let ``gather selects elements by index`` () =
    let t = torch.tensor ([| 1.0f; 2.0f; 3.0f; 4.0f; 5.0f; 6.0f |], device = torch.CPU)


    let t = t.reshape [| 2L; 3L |]
    let idx = torch.zeros ([| 2L; 1L |], dtype = torch.int64, device = torch.CPU)
    let g = t.gather (1L, idx)
    g.shape |> should equal [| 2L; 1L |]

[<Fact>]
let ``indexSelect selects rows`` () =
    let t =
        torch.tensor ([| 10.0f; 20.0f; 30.0f; 40.0f; 50.0f; 60.0f |], device = torch.CPU)


    let t = t.reshape [| 3L; 2L |]
    let idx = torch.zeros ([| 1L |], dtype = torch.int64, device = torch.CPU)
    let s = t.index_select (0L, idx)
    s.shape |> should equal [| 1L; 2L |]
    scalarF32 s |> should equal 30.0f

[<Fact>]
let ``clamp constrains values`` () =
    let t = torch.tensor ([| -5.0f; 0.0f; 5.0f |], device = torch.CPU)

    let c = t.clamp (min = scalar (-1.0), max = scalar 1.0)
    scalarF32 c |> should equal 0.0f

[<Fact>]
let ``affine applies linear transform`` () =
    let t = torch.ones ([| 3L |], dtype = torch.float32, device = torch.CPU)
    let a = t.mul(scalar 2.0).add (scalar 3.0)
    scalarF32 a |> should equal 15.0f

[<Fact>]
let ``softmax produces valid distribution`` () =
    let t = torch.randn ([| 2L; 5L |], dtype = torch.float32, device = torch.CPU)
    let s = torch.nn.functional.softmax (t, 1L)

    s.shape |> should equal [| 2L; 5L |]

    let sums = s.sum ([| 1L |])
    let sumVal = (sums.sum ()).ToSingle()
    sumVal |> should (equalWithin 1e-4f) 2.0f

[<Fact>]
let ``logSoftmax produces log probabilities`` () =
    let t = torch.randn ([| 2L; 5L |], dtype = torch.float32, device = torch.CPU)
    let ls = torch.nn.functional.log_softmax (t, -1L)
    ls.shape |> should equal [| 2L; 5L |]

    let expLs = ls.exp ()
    let sums = expLs.sum ([| 1L |])
    let total = (sums.sum ()).ToSingle()
    total |> should (equalWithin 1e-4f) 2.0f

[<Fact>]
let ``operators work correctly`` () =
    let a = torch.ones ([| 2L; 2L |], dtype = torch.float32, device = torch.CPU)
    let b = torch.ones ([| 2L; 2L |], dtype = torch.float32, device = torch.CPU)

    scalarF32 (a + b) |> should equal 8.0f

[<Fact>]
let ``scalar arithmetic works`` () =
    let t = torch.ones ([| 3L |], dtype = torch.float32, device = torch.CPU)
    scalarF32 (t * 5.0) |> should equal 15.0f

[<Fact>]
let ``cat concatenates tensors`` () =
    let a = torch.ones ([| 2L; 3L |], dtype = torch.float32, device = torch.CPU)
    let b = torch.ones ([| 2L; 3L |], dtype = torch.float32, device = torch.CPU)

    let c = torch.cat ([| a; b |], dim = 0L)
    c.shape |> should equal [| 4L; 3L |]

[<Fact>]
let ``stack stacks tensors along new dim`` () =
    let a = torch.ones ([| 2L; 3L |], dtype = torch.float32, device = torch.CPU)
    let b = torch.ones ([| 2L; 3L |], dtype = torch.float32, device = torch.CPU)

    let s = torch.stack ([| a; b |], dim = 0L)
    s.shape |> should equal [| 2L; 2L; 3L |]

[<Fact>]
let ``toDType converts type`` () =
    let t = torch.ones ([| 2L |], dtype = torch.float32, device = torch.CPU)

    let t64 = t.``to`` (torch.float64)
    t64.dtype |> should equal torch.float64

[<Fact>]
let ``ofArray creates 1D tensor`` () =
    let t = torch.tensor ([| 1.0f; 2.0f; 3.0f |], device = torch.CPU)

    t.shape |> should equal [| 3L |]
    scalarF32 t |> should equal 6.0f

[<Fact>]
let ``arange creates range tensor`` () =
    let t = torch.arange (scalar 5.0, dtype = torch.float32, device = torch.CPU)

    t.shape |> should equal [| 5L |]
    scalarF32 t |> should equal 10.0f

[<Fact>]
let ``rand creates tensor in 0-1 range`` () =
    let t = torch.rand ([| 10000L |], dtype = torch.float64, device = torch.CPU)
    let mean = scalarF64 t / 10000.0

    mean |> should be (greaterThan 0.3)
    mean |> should be (lessThan 0.7)

[<Fact>]
let ``Tensor implements IDisposable`` () =
    let t = torch.ones ([| 2L; 3L |], dtype = torch.float32, device = torch.CPU)
    t.Dispose()

[<Fact>]
let ``chained operations produce correct shape`` () =
    let a = torch.randn ([| 2L; 3L |], dtype = torch.float32, device = torch.CPU)
    let b = torch.randn ([| 3L; 4L |], dtype = torch.float32, device = torch.CPU)
    let c = a.matmul b
    c.shape |> should equal [| 2L; 4L |]

[<Fact>]
let ``for loop accumulates tensor values`` () =
    let t = torch.zeros ([| 3L |], dtype = torch.float32, device = torch.CPU)
    let mutable acc = t

    for _ in 1..4 do
        let ones = torch.ones ([| 3L |], dtype = torch.float32, device = torch.CPU)
        acc <- acc + ones

    scalarF32 acc |> should equal 12.0f

[<Fact>]
let ``use disposes resource after scope`` () =
    let disposed = ref false

    let shape =
        use _d =
            { new System.IDisposable with
                member _.Dispose() = disposed.Value <- true
            }

        let t = torch.ones ([| 2L |], dtype = torch.float32, device = torch.CPU)
        t.shape

    shape |> should equal [| 2L |]
    disposed.Value |> should be True

[<Fact>]
let ``use disposes tensor after scope`` () =
    let mutable innerRef: TorchSharp.torch.Tensor = null

    let shape =
        use t = torch.ones ([| 4L |], dtype = torch.float32, device = torch.CPU)
        innerRef <- t
        t.shape

    shape |> should equal [| 4L |]
    innerRef.IsInvalid |> should equal true

[<Fact>]
let ``oneHot encodes integer tensor`` () =
    let t = torch.tensor ([| 0.0f; 1.0f; 2.0f |], device = torch.CPU)

    let t = t.``to`` (torch.int64)
    let oh = torch.nn.functional.one_hot(t, 3L).``to`` (torch.float32)
    oh.shape |> should equal [| 3L; 3L |]

    let sum = (oh.sum ()).ToSingle()
    sum |> should equal 3.0f

[<Fact>]
let ``chunk splits tensor along dimension`` () =
    let t = torch.ones ([| 2L; 12L |], dtype = torch.float32, device = torch.CPU)
    let chunks = t.chunk (4L, 1L)
    chunks.Length |> should equal 4
    chunks[0].shape |> should equal [| 2L; 3L |]

[<Fact>]
let ``add broadcasts automatically`` () =
    let a = torch.ones ([| 2L; 3L |], dtype = torch.float32, device = torch.CPU)
    let b = torch.ones ([| 3L |], dtype = torch.float32, device = torch.CPU)
    let c = a.add b
    c.shape |> should equal [| 2L; 3L |]
    scalarF32 c |> should equal 12.0f

[<Fact>]
let ``narrow extracts slice along dimension`` () =
    let t = torch.arange (scalar 10.0, dtype = torch.float32, device = torch.CPU)
    let t = t.reshape [| 2L; 5L |]
    let s = t.narrow (1, 1L, 3L)
    s.shape |> should equal [| 2L; 3L |]

[<Fact>]
let ``leakyRelu applies negative slope`` () =
    let t = torch.tensor ([| -2.0f; -1.0f; 0.0f; 1.0f; 2.0f |], device = torch.CPU)


    let y = torch.nn.functional.leaky_relu (t, 0.1)
    let vals = y.shape
    vals |> should equal [| 5L |]

[<Fact>]
let ``elu applies exponential linear unit`` () =
    let t = torch.tensor ([| -1.0f; 0.0f; 1.0f |], device = torch.CPU)

    let y = t.elu (1.0)
    y.shape |> should equal [| 3L |]

[<Fact>]
let ``mish preserves shape`` () =
    let t = torch.randn ([| 3L; 4L |], dtype = torch.float32, device = torch.CPU)
    let y = torch.nn.functional.mish (t)
    y.shape |> should equal [| 3L; 4L |]

[<Fact>]
let ``maxPool2d reduces spatial dimensions`` () =
    let t =
        torch.randn ([| 1L; 1L; 4L; 4L |], dtype = torch.float32, device = torch.CPU)

    let y = torch.nn.functional.max_pool2d (t, 2L)
    y.shape |> should equal [| 1L; 1L; 2L; 2L |]

[<Fact>]
let ``avgPool2d reduces spatial dimensions`` () =
    let t =
        torch.randn ([| 1L; 1L; 4L; 4L |], dtype = torch.float32, device = torch.CPU)

    let y = torch.nn.functional.avg_pool2d (t, 2L)
    y.shape |> should equal [| 1L; 1L; 2L; 2L |]

[<Fact>]
let ``maskedFill replaces masked positions`` () =
    let t = torch.ones ([| 2L; 3L |], dtype = torch.float32, device = torch.CPU)

    let boolMask =
        TorchSharp.torch.tensor (array2D [| [| false; true; false |]; [| true; false; true |] |])



    let filled = t.masked_fill (boolMask, scalar -999.0)
    filled.shape |> should equal [| 2L; 3L |]

[<Fact>]
let ``Item int selects along dim 0`` () =
    let t = torch.arange (scalar 6.0, dtype = torch.float32, device = torch.CPU)
    let t = t.reshape [| 3L; 2L |]
    let row = t[0]
    row.shape |> should equal [| 2L |]
    scalarF32 row |> should equal 1.0f

[<Fact>]
let ``Item Tensor selects rows by index`` () =
    let t = torch.arange (scalar 6.0, dtype = torch.float32, device = torch.CPU)
    let t = t.reshape [| 3L; 2L |]
    let idx = torch.tensor ([| 0.0f; 2.0f |], device = torch.CPU)
    let idx = idx.``to`` (torch.int64)
    let selected = t[idx]
    selected.shape |> should equal [| 2L; 2L |]

[<Fact>]
let ``GetSlice extracts range`` () =
    let t = torch.arange (scalar 5.0, dtype = torch.float32, device = torch.CPU)
    let s = t.at [ S(1, 4) ]
    s.shape |> should equal [| 3L |]
    scalarF32 s |> should equal 6.0f

[<Fact>]
let ``GetSlice open-ended selects to end`` () =
    let t = torch.arange (scalar 5.0, dtype = torch.float32, device = torch.CPU)
    let s = t.at [ Sf 2 ]
    s.shape |> should equal [| 3L |]

[<Fact>]
let ``GetSlice with -1 end selects all`` () =
    let t = torch.arange (scalar 5.0, dtype = torch.float32, device = torch.CPU)
    let s = t.at [ A ]
    s.shape |> should equal [| 5L |]

[<Fact>]
let ``at with TIdx selects correctly`` () =
    let t = torch.arange (scalar 12.0, dtype = torch.float32, device = torch.CPU)
    let t = t.reshape [| 3L; 4L |]
    let s = t.at [ I 1; S(0, 2) ]
    s.shape |> should equal [| 2L |]

[<Fact>]
let ``at with Tensor TIdx performs advanced indexing`` () =
    let t = torch.arange (scalar 6.0, dtype = torch.float32, device = torch.CPU)
    let t = t.reshape [| 3L; 2L |]
    let idx = torch.tensor ([| 0.0f; 2.0f |], device = torch.CPU)
    let idx = idx.``to`` (torch.int64)
    let s = t.at [ T idx ]
    s.shape |> should equal [| 2L; 2L |]

[<Fact>]
let ``at with Ellipsis selects trailing dim`` () =
    let t = torch.arange (scalar 24.0, dtype = torch.float32, device = torch.CPU)
    let t = t.reshape [| 2L; 3L; 4L |]
    let s = t.at [ E; I 0 ]
    s.shape |> should equal [| 2L; 3L |]

[<Fact>]
let ``argmax returns index of maximum along dim`` () =
    let t = torch.tensor ([| 1.0f; 3.0f; 2.0f; 5.0f; 4.0f; 0.0f |], device = torch.CPU)


    let t = t.reshape [| 2L; 3L |]
    let idx = t.argmax (1L)
    idx.shape |> should equal [| 2L |]
    idx[0].ToInt64() |> should equal 1L
    idx[1].ToInt64() |> should equal 0L

[<Fact>]
let ``eq returns bool tensor`` () =
    let a = torch.tensor ([| 1.0f; 2.0f; 3.0f |], device = torch.CPU)

    let b = torch.tensor ([| 1.0f; 0.0f; 3.0f |], device = torch.CPU)

    let eq = a.eq b
    eq.shape |> should equal [| 3L |]
    let eqSum = eq.sum ()
    eqSum.ToDouble() |> should equal 2.0

[<Fact>]
let ``item returns scalar as float`` () =
    let t =
        torch.full ([| 1L |], scalar 3.14, dtype = torch.float64, device = torch.CPU)

    t.ToDouble() |> should (equalWithin 1e-10) 3.14

[<Fact>]
let ``max returns values and indices`` () =
    let t = torch.tensor ([| 1.0f; 5.0f; 3.0f; 2.0f; 4.0f; 0.0f |], device = torch.CPU)


    let t = t.reshape [| 2L; 3L |]
    let struct (values, indices) = t.max (1L)
    values.shape |> should equal [| 2L |]
    indices.shape |> should equal [| 2L |]

[<Fact>]
let ``permute reorders dimensions`` () =
    let t = torch.randn ([| 2L; 3L; 4L |], dtype = torch.float32, device = torch.CPU)
    let p = t.permute [| 0L; 2L; 1L |]
    p.shape |> should equal [| 2L; 4L; 3L |]

[<Fact>]
let ``expand broadcasts to larger shape`` () =
    let t = torch.ones ([| 1L; 3L |], dtype = torch.float32, device = torch.CPU)
    let e = t.expand [| 4L; 3L |]
    e.shape |> should equal [| 4L; 3L |]

[<Fact>]
let ``repeatInterleave repeats elements`` () =
    let t = torch.ones ([| 2L; 3L |], dtype = torch.float32, device = torch.CPU)
    let r = t.repeat_interleave (2L, dim = 0L)
    r.shape |> should equal [| 4L; 3L |]

[<Fact>]
let ``pad adds padding to tensor`` () =
    let t = torch.ones ([| 2L; 3L |], dtype = torch.float32, device = torch.CPU)
    let p = torch.nn.functional.pad (t, [| 1L; 1L; 0L; 0L |], value = 0.0)
    p.shape |> should equal [| 2L; 5L |]

[<Fact>]
let ``tril returns lower triangular`` () =
    let t = torch.ones ([| 3L; 3L |], dtype = torch.float32, device = torch.CPU)
    let lo = t.tril ()
    lo.at([ I 0; I 0 ]).ToSingle() |> should equal 1.0f
    lo.at([ I 0; I 2 ]).ToSingle() |> should equal 0.0f

[<Fact>]
let ``triu returns upper triangular`` () =
    let t = torch.ones ([| 3L; 3L |], dtype = torch.float32, device = torch.CPU)
    let up = t.triu ()
    up.at([ I 0; I 2 ]).ToSingle() |> should equal 1.0f
    up.at([ I 2; I 0 ]).ToSingle() |> should equal 0.0f
