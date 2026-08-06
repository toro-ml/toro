module AttentionTests

open Xunit
open FsUnit.Xunit
open Toro
open Toro.NN
open TestHelper

[<Fact>]
let ``scaledDotProductAttention returns correct shape`` () =
    let batch = 2
    let heads = 4
    let seqLen = 8
    let headDim = 16

    let q = Tensor.randn ([ batch; heads; seqLen; headDim ], F32, Cpu) |> unwrap
    let k = Tensor.randn ([ batch; heads; seqLen; headDim ], F32, Cpu) |> unwrap
    let v = Tensor.randn ([ batch; heads; seqLen; headDim ], F32, Cpu) |> unwrap

    let out = q.scaledDotProductAttention (k, v) |> unwrap
    out.Shape |> should equal [ batch; heads; seqLen; headDim ]

[<Fact>]
let ``scaledDotProductAttention with causal mask`` () =
    let q = Tensor.randn ([ 1; 1; 4; 8 ], F32, Cpu) |> unwrap
    let k = Tensor.randn ([ 1; 1; 4; 8 ], F32, Cpu) |> unwrap
    let v = Tensor.randn ([ 1; 1; 4; 8 ], F32, Cpu) |> unwrap

    let out = q.scaledDotProductAttention (k, v, isCausal = true) |> unwrap
    out.Shape |> should equal [ 1; 1; 4; 8 ]

[<Fact>]
let ``scaledDotProductAttention with explicit attn mask`` () =
    let seqLen = 6
    let q = Tensor.randn ([ 1; 2; seqLen; 8 ], F32, Cpu) |> unwrap
    let k = Tensor.randn ([ 1; 2; seqLen; 8 ], F32, Cpu) |> unwrap
    let v = Tensor.randn ([ 1; 2; seqLen; 8 ], F32, Cpu) |> unwrap
    let mask = Tensor.causalMask (seqLen, F32, Cpu) |> unwrap

    let out = q.scaledDotProductAttention (k, v, attnMask = mask) |> unwrap
    out.Shape |> should equal [ 1; 2; seqLen; 8 ]

[<Fact>]
let ``causalMask is upper triangular neg-inf`` () =
    let mask = Tensor.causalMask (4, F32, Cpu) |> unwrap
    mask.Shape |> should equal [ 4; 4 ]

    let diagVal = mask.Inner.data<float32>().[(0 * 4 + 0)]
    diagVal |> should equal 0.0f

    let upperVal = mask.Inner.data<float32>().[(0 * 4 + 1)]
    upperVal |> should equal System.Single.NegativeInfinity

[<Fact>]
let ``KvCache append accumulates sequence length`` () =
    let cache = KvCache.create 2
    cache.CurrentSeqLen |> should equal 0

    let k1 = Tensor.randn ([ 1; 4; 3; 8 ], F32, Cpu) |> unwrap
    let v1 = Tensor.randn ([ 1; 4; 3; 8 ], F32, Cpu) |> unwrap
    let (ck1, cv1) = cache.append (k1, v1) |> unwrap
    cache.CurrentSeqLen |> should equal 3
    ck1.Shape |> should equal [ 1; 4; 3; 8 ]
    cv1.Shape |> should equal [ 1; 4; 3; 8 ]

    let k2 = Tensor.randn ([ 1; 4; 2; 8 ], F32, Cpu) |> unwrap
    let v2 = Tensor.randn ([ 1; 4; 2; 8 ], F32, Cpu) |> unwrap
    let (ck2, cv2) = cache.append (k2, v2) |> unwrap
    cache.CurrentSeqLen |> should equal 5
    ck2.Shape |> should equal [ 1; 4; 5; 8 ]
    cv2.Shape |> should equal [ 1; 4; 5; 8 ]

[<Fact>]
let ``KvCache reset clears state`` () =
    let cache = KvCache.create 2
    let k = Tensor.randn ([ 1; 4; 3; 8 ], F32, Cpu) |> unwrap
    let v = Tensor.randn ([ 1; 4; 3; 8 ], F32, Cpu) |> unwrap
    cache.append (k, v) |> unwrap |> ignore

    cache.CurrentSeqLen |> should equal 3
    cache.currentData () |> Option.isSome |> should equal true

    cache.reset ()
    cache.CurrentSeqLen |> should equal 0
    cache.currentData () |> should equal None
