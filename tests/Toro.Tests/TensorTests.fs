module TensorTests

open Xunit
open FsUnit.Xunit
open Toro
open TestHelper

[<Fact>]
let ``zeros creates tensor with correct shape`` () =
    let t = Tensor.zeros ([ 2; 3 ], F32, Cpu) |> unwrap

    t.Shape |> should equal [ 2; 3 ]
    t.DType |> should equal F32
    t.Device |> should equal Cpu

[<Fact>]
let ``ones creates tensor with correct shape`` () =
    let t = Tensor.ones ([ 4; 5 ], F64, Cpu) |> unwrap

    t.Shape |> should equal [ 4; 5 ]
    t.DType |> should equal F64

[<Fact>]
let ``randn creates tensor with correct shape`` () =
    let t = Tensor.randn ([ 3; 4 ], F32, Cpu) |> unwrap

    t.Shape |> should equal [ 3; 4 ]
    t.Rank |> should equal 2
    t.ElemCount |> should equal 12L

[<Fact>]
let ``full creates tensor with given value`` () =
    let t = Tensor.full ([ 2; 2 ], 3.14, F64, Cpu) |> unwrap

    t.Shape |> should equal [ 2; 2 ]
    t.DType |> should equal F64
    scalarF64 t |> should (equalWithin 1e-10) (3.14 * 4.0)

[<Fact>]
let ``matmul produces correct shape`` () =
    let a = Tensor.randn ([ 2; 3 ], F32, Cpu) |> unwrap
    let b = Tensor.randn ([ 3; 4 ], F32, Cpu) |> unwrap

    let c = a.matmul b |> unwrap
    c.Shape |> should equal [ 2; 4 ]

[<Fact>]
let ``add and sub are consistent`` () =
    let a = Tensor.ones ([ 2; 2 ], F32, Cpu) |> unwrap
    let b = Tensor.ones ([ 2; 2 ], F32, Cpu) |> unwrap

    let diff = (a.add b |> unwrap).sub b |> unwrap
    scalarF32 diff |> should equal 4.0f

[<Fact>]
let ``div divides tensors element-wise`` () =
    let a = Tensor.full ([ 3 ], 6.0, F32, Cpu) |> unwrap
    let b = Tensor.full ([ 3 ], 2.0, F32, Cpu) |> unwrap
    let c = a.div b |> unwrap
    scalarF32 c |> should equal 9.0f

[<Fact>]
let ``divScalar divides by scalar`` () =
    let t = Tensor.full ([ 3 ], 6.0, F32, Cpu) |> unwrap
    let s = t.divScalar 3.0 |> unwrap
    scalarF32 s |> should equal 6.0f

[<Fact>]
let ``subScalar subtracts scalar value`` () =
    let t = Tensor.full ([ 3 ], 5.0, F32, Cpu) |> unwrap
    let s = t.subScalar 2.0 |> unwrap
    scalarF32 s |> should equal 9.0f

[<Fact>]
let ``pow computes element-wise power`` () =
    let t = Tensor.full ([ 3 ], 2.0, F64, Cpu) |> unwrap
    let p = t.pow 3.0 |> unwrap
    scalarF64 p |> should (equalWithin 1e-10) 24.0

[<Fact>]
let ``reshape changes shape`` () =
    let t = Tensor.randn ([ 2; 3 ], F32, Cpu) |> unwrap

    let r = t.reshape [ 6 ] |> unwrap
    r.Shape |> should equal [ 6 ]
    r.Rank |> should equal 1

[<Fact>]
let ``view reshapes tensor`` () =
    let t = Tensor.randn ([ 2; 6 ], F32, Cpu) |> unwrap
    let v = t.view [ 3; 4 ] |> unwrap
    v.Shape |> should equal [ 3; 4 ]

[<Fact>]
let ``flatten collapses dimensions`` () =
    let t = Tensor.randn ([ 2; 3; 4 ], F32, Cpu) |> unwrap
    let f = t.flatten (1, 2) |> unwrap
    f.Shape |> should equal [ 2; 12 ]

[<Fact>]
let ``flattenAll creates 1D tensor`` () =
    let t = Tensor.randn ([ 2; 3; 4 ], F32, Cpu) |> unwrap
    let f = t.flattenAll () |> unwrap
    f.Shape |> should equal [ 24 ]

[<Fact>]
let ``transpose swaps dimensions`` () =
    let t = Tensor.randn ([ 2; 3 ], F32, Cpu) |> unwrap

    let tr = t.t () |> unwrap
    tr.Shape |> should equal [ 3; 2 ]

[<Fact>]
let ``squeeze removes dim of size 1`` () =
    let t = Tensor.ones ([ 2; 1; 3 ], F32, Cpu) |> unwrap
    let s = t.squeeze 1 |> unwrap
    s.Shape |> should equal [ 2; 3 ]

[<Fact>]
let ``unsqueeze adds dim of size 1`` () =
    let t = Tensor.ones ([ 2; 3 ], F32, Cpu) |> unwrap
    let u = t.unsqueeze 1 |> unwrap
    u.Shape |> should equal [ 2; 1; 3 ]

[<Fact>]
let ``sum keepDim preserves rank`` () =
    let t = Tensor.ones ([ 2; 3; 4 ], F32, Cpu) |> unwrap

    let s = t.sum (-1, keepDim = true) |> unwrap
    s.Shape |> should equal [ 2; 3; 1 ]

[<Fact>]
let ``unary ops preserve shape`` () =
    let t = Tensor.ones ([ 2; 3 ], F32, Cpu) |> unwrap

    (t.neg () |> unwrap).Shape |> should equal [ 2; 3 ]
    (t.abs () |> unwrap).Shape |> should equal [ 2; 3 ]
    (t.sqrt () |> unwrap).Shape |> should equal [ 2; 3 ]
    (t.exp () |> unwrap).Shape |> should equal [ 2; 3 ]
    (t.log () |> unwrap).Shape |> should equal [ 2; 3 ]

[<Fact>]
let ``neg negates values`` () =
    let t = Tensor.ones ([ 3 ], F32, Cpu) |> unwrap
    let n = t.neg () |> unwrap
    scalarF32 n |> should equal -3.0f

[<Fact>]
let ``clone creates independent copy`` () =
    let t = Tensor.ones ([ 3 ], F32, Cpu) |> unwrap
    let c = t.clone () |> unwrap
    c.Shape |> should equal [ 3 ]
    scalarF32 c |> should equal 3.0f

[<Fact>]
let ``gather selects elements by index`` () =
    let t =
        Tensor.ofFloat32Array ([| 1.0f; 2.0f; 3.0f; 4.0f; 5.0f; 6.0f |], Cpu)
        |> unwrap

    let t = t.reshape [ 2; 3 ] |> unwrap
    let idx = Tensor.zeros ([ 2; 1 ], I64, Cpu) |> unwrap
    let g = t.gather (1, idx) |> unwrap
    g.Shape |> should equal [ 2; 1 ]

[<Fact>]
let ``indexSelect selects rows`` () =
    let t =
        Tensor.ofFloat32Array ([| 10.0f; 20.0f; 30.0f; 40.0f; 50.0f; 60.0f |], Cpu)
        |> unwrap

    let t = t.reshape [ 3; 2 ] |> unwrap
    let idx = Tensor.zeros ([ 1 ], I64, Cpu) |> unwrap
    let s = t.indexSelect (0, idx) |> unwrap
    s.Shape |> should equal [ 1; 2 ]
    scalarF32 s |> should equal 30.0f

[<Fact>]
let ``clamp constrains values`` () =
    let t =
        Tensor.ofFloat32Array ([| -5.0f; 0.0f; 5.0f |], Cpu)
        |> unwrap

    let c = t.clamp (-1.0, 1.0) |> unwrap
    scalarF32 c |> should equal 0.0f

[<Fact>]
let ``affine applies linear transform`` () =
    let t = Tensor.ones ([ 3 ], F32, Cpu) |> unwrap
    let a = t.affine (2.0, 3.0) |> unwrap
    scalarF32 a |> should equal 15.0f

[<Fact>]
let ``softmax produces valid distribution`` () =
    let t = Tensor.randn ([ 2; 5 ], F32, Cpu) |> unwrap
    let s = t.softmax 1 |> unwrap

    s.Shape |> should equal [ 2; 5 ]

    let sums = s.sum (1) |> unwrap
    let sumVal = (sums.sumAll () |> unwrap).toFloat32Scalar () |> unwrap
    sumVal |> should (equalWithin 1e-4f) 2.0f

[<Fact>]
let ``logSoftmax produces log probabilities`` () =
    let t = Tensor.randn ([ 2; 5 ], F32, Cpu) |> unwrap
    let ls = t.logSoftmax -1 |> unwrap
    ls.Shape |> should equal [ 2; 5 ]

    let expLs = ls.exp () |> unwrap
    let sums = expLs.sum (1) |> unwrap
    let total = (sums.sumAll () |> unwrap).toFloat32Scalar () |> unwrap
    total |> should (equalWithin 1e-4f) 2.0f

[<Fact>]
let ``operators work correctly`` () =
    let a = Tensor.ones ([ 2; 2 ], F32, Cpu) |> unwrap
    let b = Tensor.ones ([ 2; 2 ], F32, Cpu) |> unwrap

    scalarF32 (a + b) |> should equal 8.0f

[<Fact>]
let ``scalar arithmetic works`` () =
    let t = Tensor.ones ([ 3 ], F32, Cpu) |> unwrap
    scalarF32 (t * 5.0) |> should equal 15.0f

[<Fact>]
let ``cat concatenates tensors`` () =
    let a = Tensor.ones ([ 2; 3 ], F32, Cpu) |> unwrap
    let b = Tensor.ones ([ 2; 3 ], F32, Cpu) |> unwrap

    let c = Tensor.cat ([ a; b ], 0) |> unwrap
    c.Shape |> should equal [ 4; 3 ]

[<Fact>]
let ``stack stacks tensors along new dim`` () =
    let a = Tensor.ones ([ 2; 3 ], F32, Cpu) |> unwrap
    let b = Tensor.ones ([ 2; 3 ], F32, Cpu) |> unwrap

    let s = Tensor.stack ([ a; b ], 0) |> unwrap
    s.Shape |> should equal [ 2; 2; 3 ]

[<Fact>]
let ``toDType converts type`` () =
    let t = Tensor.ones ([ 2 ], F32, Cpu) |> unwrap

    let t64 = t.toDType F64 |> unwrap
    t64.DType |> should equal F64

[<Fact>]
let ``ofFloat32Array creates 1D tensor`` () =
    let t =
        Tensor.ofFloat32Array ([| 1.0f; 2.0f; 3.0f |], Cpu)
        |> unwrap

    t.Shape |> should equal [ 3 ]
    scalarF32 t |> should equal 6.0f

[<Fact>]
let ``arange creates range tensor`` () =
    let t = Tensor.arange (5.0, F32, Cpu) |> unwrap

    t.Shape |> should equal [ 5 ]
    scalarF32 t |> should equal 10.0f

[<Fact>]
let ``rand creates tensor in 0-1 range`` () =
    let t = Tensor.rand ([ 10000 ], F64, Cpu) |> unwrap
    let mean = scalarF64 t / 10000.0

    mean |> should be (greaterThan 0.3)
    mean |> should be (lessThan 0.7)

[<Fact>]
let ``Tensor implements IDisposable`` () =
    let t = Tensor.ones ([ 2; 3 ], F32, Cpu) |> unwrap
    t.Dispose()

[<Fact>]
let ``result CE chains operations`` () =
    let r =
        result {
            let! a = Tensor.randn ([ 2; 3 ], F32, Cpu)
            let! b = Tensor.randn ([ 3; 4 ], F32, Cpu)
            let! c = a.matmul b
            return c.Shape
        }

    unwrap r |> should equal [ 2; 4 ]

[<Fact>]
let ``result CE supports for loop`` () =
    let r =
        result {
            let! t = Tensor.zeros ([ 3 ], F32, Cpu)
            let mutable acc = t

            for _ in 1..4 do
                let! ones = Tensor.ones ([ 3 ], F32, Cpu)
                acc <- acc + ones

            return acc
        }

    let t = unwrap r
    scalarF32 t |> should equal 12.0f

[<Fact>]
let ``result CE supports try-with`` () =
    let r: Result<int, ToroError> =
        result {
            try
                return! Error(Msg "test error")
            with _ ->
                return 42
        }

    match r with
    | Error(Msg "test error") -> ()
    | other -> failwithf "Expected Msg error, got: %A" other

[<Fact>]
let ``result CE supports use`` () =
    let disposed = ref false

    let r =
        result {
            use _d =
                { new System.IDisposable with
                    member _.Dispose() = disposed.Value <- true
                }

            let! t = Tensor.ones ([ 2 ], F32, Cpu)
            return t.Shape
        }

    unwrap r |> should equal [ 2 ]
    disposed.Value |> should be True

[<Fact>]
let ``use! disposes tensor after scope`` () =
    let mutable innerRef: TorchSharp.torch.Tensor = null

    let r =
        result {
            use! t = Tensor.ones ([ 4 ], F32, Cpu)
            innerRef <- t.Inner
            return t.Shape
        }

    unwrap r |> should equal [ 4 ]

[<Fact>]
let ``oneHot encodes integer tensor`` () =
    let t =
        Tensor.ofFloat32Array ([| 0.0f; 1.0f; 2.0f |], Cpu)
        |> unwrap

    let t = t.toDType I64 |> unwrap
    let oh = t.oneHot 3 |> unwrap
    oh.Shape |> should equal [ 3; 3 ]

    let sum = (oh.sumAll () |> unwrap).toFloat32Scalar () |> unwrap
    sum |> should equal 3.0f

[<Fact>]
let ``chunk splits tensor along dimension`` () =
    let t = Tensor.ones ([ 2; 12 ], F32, Cpu) |> unwrap
    let chunks = t.chunk (4, 1) |> unwrap
    chunks.Length |> should equal 4
    chunks[0].Shape |> should equal [ 2; 3 ]

[<Fact>]
let ``add broadcasts automatically`` () =
    let a = Tensor.ones ([ 2; 3 ], F32, Cpu) |> unwrap
    let b = Tensor.ones ([ 3 ], F32, Cpu) |> unwrap
    let c = a.add b |> unwrap
    c.Shape |> should equal [ 2; 3 ]
    scalarF32 c |> should equal 12.0f

[<Fact>]
let ``narrow extracts slice along dimension`` () =
    let t = Tensor.arange (10.0, F32, Cpu) |> unwrap
    let t = t.reshape [ 2; 5 ] |> unwrap
    let s = t.narrow (1, 1L, 3L) |> unwrap
    s.Shape |> should equal [ 2; 3 ]

[<Fact>]
let ``leakyRelu applies negative slope`` () =
    let t =
        Tensor.ofFloat32Array ([| -2.0f; -1.0f; 0.0f; 1.0f; 2.0f |], Cpu)
        |> unwrap

    let y = t.leakyRelu 0.1 |> unwrap
    let vals = y.Shape
    vals |> should equal [ 5 ]

[<Fact>]
let ``elu applies exponential linear unit`` () =
    let t =
        Tensor.ofFloat32Array ([| -1.0f; 0.0f; 1.0f |], Cpu)
        |> unwrap

    let y = t.elu 1.0 |> unwrap
    y.Shape |> should equal [ 3 ]

[<Fact>]
let ``mish preserves shape`` () =
    let t = Tensor.randn ([ 3; 4 ], F32, Cpu) |> unwrap
    let y = t.mish () |> unwrap
    y.Shape |> should equal [ 3; 4 ]

[<Fact>]
let ``maxPool2d reduces spatial dimensions`` () =
    let t = Tensor.randn ([ 1; 1; 4; 4 ], F32, Cpu) |> unwrap
    let y = t.maxPool2d 2 |> unwrap
    y.Shape |> should equal [ 1; 1; 2; 2 ]

[<Fact>]
let ``avgPool2d reduces spatial dimensions`` () =
    let t = Tensor.randn ([ 1; 1; 4; 4 ], F32, Cpu) |> unwrap
    let y = t.avgPool2d 2 |> unwrap
    y.Shape |> should equal [ 1; 1; 2; 2 ]

[<Fact>]
let ``maskedFill replaces masked positions`` () =
    let t = Tensor.ones ([ 2; 3 ], F32, Cpu) |> unwrap

    let boolMask =
        TorchSharp.torch.tensor (array2D [| [| false; true; false |]; [| true; false; true |] |])
        |> Tensor.ofTorchTensor
        |> unwrap

    let filled = t.maskedFill (boolMask, -999.0) |> unwrap
    filled.Shape |> should equal [ 2; 3 ]

[<Fact>]
let ``Item int selects along dim 0`` () =
    let t = Tensor.arange (6.0, F32, Cpu) |> unwrap
    let t = t.reshape [ 3; 2 ] |> unwrap
    let row = t[0]
    row.Shape |> should equal [ 2 ]
    scalarF32 row |> should equal 1.0f

[<Fact>]
let ``Item Tensor selects rows by index`` () =
    let t = Tensor.arange (6.0, F32, Cpu) |> unwrap
    let t = t.reshape [ 3; 2 ] |> unwrap
    let idx = Tensor.ofFloat32Array ([| 0.0f; 2.0f |], Cpu) |> unwrap
    let idx = idx.toDType I64 |> unwrap
    let selected = t[idx]
    selected.Shape |> should equal [ 2; 2 ]

[<Fact>]
let ``GetSlice extracts range`` () =
    let t = Tensor.arange (5.0, F32, Cpu) |> unwrap
    let s = t[1..3]
    s.Shape |> should equal [ 3 ]
    scalarF32 s |> should equal 6.0f

[<Fact>]
let ``GetSlice open-ended selects to end`` () =
    let t = Tensor.arange (5.0, F32, Cpu) |> unwrap
    let s = t[2..]
    s.Shape |> should equal [ 3 ]

[<Fact>]
let ``GetSlice with -1 end selects all`` () =
    let t = Tensor.arange (5.0, F32, Cpu) |> unwrap
    let s = t[0 .. -1]
    s.Shape |> should equal [ 5 ]

[<Fact>]
let ``at with TIdx selects correctly`` () =
    let t = Tensor.arange (12.0, F32, Cpu) |> unwrap
    let t = t.reshape [ 3; 4 ] |> unwrap
    let s = t.at [ I 1; S(0, 2) ]
    s.Shape |> should equal [ 2 ]

[<Fact>]
let ``at with Tensor TIdx performs advanced indexing`` () =
    let t = Tensor.arange (6.0, F32, Cpu) |> unwrap
    let t = t.reshape [ 3; 2 ] |> unwrap
    let idx = Tensor.ofFloat32Array ([| 0.0f; 2.0f |], Cpu) |> unwrap
    let idx = idx.toDType I64 |> unwrap
    let s = t.at [ T idx ]
    s.Shape |> should equal [ 2; 2 ]

[<Fact>]
let ``at with Ellipsis selects trailing dim`` () =
    let t = Tensor.arange (24.0, F32, Cpu) |> unwrap
    let t = t.reshape [ 2; 3; 4 ] |> unwrap
    let s = t.at [ E; I 0 ]
    s.Shape |> should equal [ 2; 3 ]

[<Fact>]
let ``argmax returns index of maximum along dim`` () =
    let t =
        Tensor.ofFloat32Array ([| 1.0f; 3.0f; 2.0f; 5.0f; 4.0f; 0.0f |], Cpu)
        |> unwrap

    let t = t.reshape [ 2; 3 ] |> unwrap
    let idx = t.argmax 1 |> unwrap
    idx.Shape |> should equal [ 2 ]
    idx[0].itemI64 () |> should equal 1L
    idx[1].itemI64 () |> should equal 0L

[<Fact>]
let ``eq returns bool tensor`` () =
    let a =
        Tensor.ofFloat32Array ([| 1.0f; 2.0f; 3.0f |], Cpu)
        |> unwrap

    let b =
        Tensor.ofFloat32Array ([| 1.0f; 0.0f; 3.0f |], Cpu)
        |> unwrap

    let eq = a.eq b
    eq.Shape |> should equal [ 3 ]
    let eqSum = eq.sumAll () |> unwrap
    eqSum.item () |> should equal 2.0

[<Fact>]
let ``item returns scalar as float`` () =
    let t = Tensor.full ([ 1 ], 3.14, F64, Cpu) |> unwrap
    t.item () |> should (equalWithin 1e-10) 3.14

[<Fact>]
let ``max returns values and indices`` () =
    let t =
        Tensor.ofFloat32Array ([| 1.0f; 5.0f; 3.0f; 2.0f; 4.0f; 0.0f |], Cpu)
        |> unwrap

    let t = t.reshape [ 2; 3 ] |> unwrap
    let values, indices = t.max 1 |> unwrap
    values.Shape |> should equal [ 2 ]
    indices.Shape |> should equal [ 2 ]

[<Fact>]
let ``permute reorders dimensions`` () =
    let t = Tensor.randn ([ 2; 3; 4 ], F32, Cpu) |> unwrap
    let p = t.permute [ 0; 2; 1 ] |> unwrap
    p.Shape |> should equal [ 2; 4; 3 ]

[<Fact>]
let ``expand broadcasts to larger shape`` () =
    let t = Tensor.ones ([ 1; 3 ], F32, Cpu) |> unwrap
    let e = t.expand [ 4; 3 ] |> unwrap
    e.Shape |> should equal [ 4; 3 ]

[<Fact>]
let ``repeatInterleave repeats elements`` () =
    let t = Tensor.ones ([ 2; 3 ], F32, Cpu) |> unwrap
    let r = t.repeatInterleave (2, 0) |> unwrap
    r.Shape |> should equal [ 4; 3 ]

[<Fact>]
let ``pad adds padding to tensor`` () =
    let t = Tensor.ones ([ 2; 3 ], F32, Cpu) |> unwrap
    let p = t.pad ([ 1; 1; 0; 0 ], 0.0) |> unwrap
    p.Shape |> should equal [ 2; 5 ]

[<Fact>]
let ``tril returns lower triangular`` () =
    let t = Tensor.ones ([ 3; 3 ], F32, Cpu) |> unwrap
    let lo = t.tril () |> unwrap
    lo.at([ I 0; I 0 ]).itemF32 () |> should equal 1.0f
    lo.at([ I 0; I 2 ]).itemF32 () |> should equal 0.0f

[<Fact>]
let ``triu returns upper triangular`` () =
    let t = Tensor.ones ([ 3; 3 ], F32, Cpu) |> unwrap
    let up = t.triu () |> unwrap
    up.at([ I 0; I 2 ]).itemF32 () |> should equal 1.0f
    up.at([ I 2; I 0 ]).itemF32 () |> should equal 0.0f
