module AttentionTests

open Xunit
open FsUnit.Xunit
open Toro
open TorchSharp
open Toro.NN
open TestHelper

let private causalMask (n: int) (dtype: torch.ScalarType) (device: torch.Device) =
    let nL = int64 n

    let upper =
        torch.triu(torch.ones ([| nL; nL |], dtype = dtype, device = device), diagonal = 1L).to_type (torch.ScalarType.Bool)

    let mask = torch.zeros ([| nL; nL |], dtype = dtype, device = device)
    mask.masked_fill (upper, scalar System.Double.NegativeInfinity)

[<Fact>]
let ``scaledDotProductAttention returns correct shape`` () =
    let batch = 2
    let heads = 4
    let seqLen = 8
    let headDim = 16

    let q =
        torch.randn ([| int64 batch; int64 heads; int64 seqLen; int64 headDim |], dtype = torch.float32, device = torch.CPU)

    let k =
        torch.randn ([| int64 batch; int64 heads; int64 seqLen; int64 headDim |], dtype = torch.float32, device = torch.CPU)

    let v =
        torch.randn ([| int64 batch; int64 heads; int64 seqLen; int64 headDim |], dtype = torch.float32, device = torch.CPU)

    let out = torch.nn.functional.scaled_dot_product_attention (q, k, v)

    out.shape
    |> should equal [| int64 batch; int64 heads; int64 seqLen; int64 headDim |]

[<Fact>]
let ``scaledDotProductAttention with causal mask`` () =
    let q =
        torch.randn ([| 1L; 1L; 4L; 8L |], dtype = torch.float32, device = torch.CPU)

    let k =
        torch.randn ([| 1L; 1L; 4L; 8L |], dtype = torch.float32, device = torch.CPU)

    let v =
        torch.randn ([| 1L; 1L; 4L; 8L |], dtype = torch.float32, device = torch.CPU)

    let out =
        torch.nn.functional.scaled_dot_product_attention (q, k, v, is_casual = true)

    out.shape |> should equal [| 1L; 1L; 4L; 8L |]

[<Fact>]
let ``scaledDotProductAttention with explicit attn mask`` () =
    let seqLen = 2
    let q = torch.randn ([| 1L; 2L; 8L |], dtype = torch.float32, device = torch.CPU)
    let k = torch.randn ([| 1L; 2L; 8L |], dtype = torch.float32, device = torch.CPU)
    let v = torch.randn ([| 1L; 2L; 8L |], dtype = torch.float32, device = torch.CPU)
    let mask = causalMask seqLen torch.float32 torch.CPU

    let out =
        torch.nn.functional.scaled_dot_product_attention (q, k, v, attn_mask = mask)

    out.shape |> should equal [| 1L; 2L; 8L |]

[<Fact>]
let ``causalMask is upper triangular neg-inf`` () =
    let mask = causalMask 4 torch.float32 torch.CPU
    mask.shape |> should equal [| 4L; 4L |]

    let diagVal = mask.data<float32>().[(0 * 4 + 0)]
    diagVal |> should equal 0.0f

    let upperVal = mask.data<float32>().[(0 * 4 + 1)]
    upperVal |> should equal System.Single.NegativeInfinity

[<Fact>]
let ``KvCache append accumulates sequence length`` () =
    let cache = KvCache.create 2
    cache.CurrentSeqLen |> should equal 0L

    let k1 =
        torch.randn ([| 1L; 4L; 3L; 8L |], dtype = torch.float32, device = torch.CPU)

    let v1 =
        torch.randn ([| 1L; 4L; 3L; 8L |], dtype = torch.float32, device = torch.CPU)

    let (ck1, cv1) = cache.append (k1, v1)
    cache.CurrentSeqLen |> should equal 3L
    ck1.shape |> should equal [| 1L; 4L; 3L; 8L |]
    cv1.shape |> should equal [| 1L; 4L; 3L; 8L |]

    let k2 =
        torch.randn ([| 1L; 4L; 2L; 8L |], dtype = torch.float32, device = torch.CPU)

    let v2 =
        torch.randn ([| 1L; 4L; 2L; 8L |], dtype = torch.float32, device = torch.CPU)

    let (ck2, cv2) = cache.append (k2, v2)
    cache.CurrentSeqLen |> should equal 5L
    ck2.shape |> should equal [| 1L; 4L; 5L; 8L |]
    cv2.shape |> should equal [| 1L; 4L; 5L; 8L |]

[<Fact>]
let ``KvCache reset clears state`` () =
    let cache = KvCache.create 2

    let k =
        torch.randn ([| 1L; 4L; 3L; 8L |], dtype = torch.float32, device = torch.CPU)

    let v =
        torch.randn ([| 1L; 4L; 3L; 8L |], dtype = torch.float32, device = torch.CPU)

    cache.append (k, v) |> ignore

    cache.CurrentSeqLen |> should equal 3L
    cache.currentData () |> Option.isSome |> should equal true

    cache.reset ()
    cache.CurrentSeqLen |> should equal 0L
    cache.currentData () |> should equal None

[<Fact>]
let ``MultiHeadAttention forward returns correct shape`` () =
    let dim = 32L
    let heads = 4
    let batch = 2L
    let seqLen = 8L
    let mha = MultiHeadAttention.init dim heads torch.float32 torch.CPU

    let x =
        torch.randn ([| batch; seqLen; dim |], dtype = torch.float32, device = torch.CPU)

    let y = mha.forward x

    y.shape |> should equal [| batch; seqLen; dim |]

[<Fact>]
let ``TransformerBlock forward returns correct shape`` () =
    let dim = 32L
    let heads = 4L
    let ffDim = 64L
    let batch = 2L
    let seqLen = 8L
    let block = TransformerBlock.init dim heads ffDim torch.float32 torch.CPU

    let x =
        torch.randn ([| batch; seqLen; dim |], dtype = torch.float32, device = torch.CPU)

    let y = block.forward x

    y.shape |> should equal [| batch; seqLen; dim |]
