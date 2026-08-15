module ScopedTests

open Xunit
open FsUnit.Xunit
open TorchSharp
open Toro
open TestHelper

// ── Test helper types ──

type AttentionOutput = { Attn: Tensor; Weights: Tensor }

type NestedRecord = {
    Output: AttentionOutput
    Loss: Tensor
}

type OptionalRecord = { Value: Tensor; Mask: Tensor option }

type LayerOutput =
    | Logits of Tensor
    | Features of k: Tensor * v: Tensor

// ═══════════════════════════════════════════════════════════════
// A. keepTensors: return value auto-keep
// ═══════════════════════════════════════════════════════════════

// A1
[<Fact>]
let ``A1 scoped keeps single tensor`` () =
    let r =
        scoped {
            let t = torch.randn ([| 4L |], dtype = torch.float32, device = torch.CPU)
            return t
        }

    let t = r
    t.shape |> should equal [| 4L |]
    t.IsInvalid |> should equal false

// A2
[<Fact>]
let ``A2 scoped with unit return`` () =
    let mutable flag = false

    let r =
        scoped {
            let _ = torch.ones ([| 2L |], dtype = torch.float32, device = torch.CPU)
            flag <- true
        }

    r |> ignore
    flag |> should equal true

// A3
[<Fact>]
let ``A3 scoped keeps 2-tuple of tensors`` () =
    let r =
        scoped {
            let a = torch.zeros ([| 3L |], dtype = torch.float32, device = torch.CPU)
            let b = torch.ones ([| 3L |], dtype = torch.float32, device = torch.CPU)
            return a, b
        }

    let a, b = r
    a.IsInvalid |> should equal false
    b.IsInvalid |> should equal false
    scalarF32 a |> should (equalWithin 1e-5f) 0.0f
    scalarF32 b |> should (equalWithin 1e-5f) 3.0f

// A4
[<Fact>]
let ``A4 scoped keeps 3-tuple of tensors`` () =
    let r =
        scoped {
            let a = torch.zeros ([| 2L |], dtype = torch.float32, device = torch.CPU)
            let b = torch.ones ([| 2L |], dtype = torch.float32, device = torch.CPU)
            let c = a.add b
            return a, b, c
        }

    let a, b, c = r
    a.IsInvalid |> should equal false
    b.IsInvalid |> should equal false
    c.IsInvalid |> should equal false
    scalarF32 c |> should (equalWithin 1e-5f) 2.0f

// A5
[<Fact>]
let ``A5 scoped keeps mixed tuple`` () =
    let r =
        scoped {
            let t = torch.ones ([| 3L |], dtype = torch.float32, device = torch.CPU)
            return t, 42
        }

    let tensor, n = r
    tensor.IsInvalid |> should equal false
    scalarF32 tensor |> should (equalWithin 1e-5f) 3.0f
    n |> should equal 42

// A6
[<Fact>]
let ``A6 scoped keeps record`` () =
    let mutable intermediate = Unchecked.defaultof<torch.Tensor>

    let r =
        scoped {
            let a = torch.ones ([| 2L; 3L |], dtype = torch.float32, device = torch.CPU)
            let w = torch.zeros ([| 2L; 2L |], dtype = torch.float32, device = torch.CPU)
            let tmp = torch.ones ([| 1L |], dtype = torch.float32, device = torch.CPU)
            intermediate <- tmp
            return { Attn = a; Weights = w }
        }

    let out = r
    out.Attn.IsInvalid |> should equal false
    out.Weights.IsInvalid |> should equal false
    intermediate.IsInvalid |> should equal true

// A7
[<Fact>]
let ``A7 scoped keeps nested record`` () =
    let r =
        scoped {
            let a = torch.ones ([| 2L |], dtype = torch.float32, device = torch.CPU)
            let w = torch.zeros ([| 2L |], dtype = torch.float32, device = torch.CPU)
            let l = torch.ones ([| 1L |], dtype = torch.float32, device = torch.CPU)

            return {
                Output = { Attn = a; Weights = w }
                Loss = l
            }
        }

    let out = r
    out.Output.Attn.IsInvalid |> should equal false
    out.Output.Weights.IsInvalid |> should equal false
    out.Loss.IsInvalid |> should equal false

// A8
[<Fact>]
let ``A8 scoped keeps tensor list`` () =
    let r =
        scoped {
            let a = torch.zeros ([| 2L |], dtype = torch.float32, device = torch.CPU)
            let b = torch.ones ([| 2L |], dtype = torch.float32, device = torch.CPU)
            return [ a; b ]
        }

    let ts = r
    ts |> List.length |> should equal 2

    ts |> List.iter (fun t -> t.IsInvalid |> should equal false)

    scalarF32 ts[1] |> should (equalWithin 1e-5f) 2.0f

// A9
[<Fact>]
let ``A9 scoped keeps tensor array`` () =
    let r =
        scoped {
            let a = torch.ones ([| 2L |], dtype = torch.float32, device = torch.CPU)
            let b = torch.ones ([| 2L |], dtype = torch.float32, device = torch.CPU)
            return [| a; b |]
        }

    let ts = r
    ts |> Array.length |> should equal 2

    ts
    |> Array.iter (fun t -> t.IsInvalid |> should equal false)

// A10
[<Fact>]
let ``A10 scoped keeps Some tensor`` () =
    let r =
        scoped {
            let t = torch.ones ([| 3L |], dtype = torch.float32, device = torch.CPU)
            return Some t
        }

    let v = r
    v.IsSome |> should equal true
    v.Value.IsInvalid |> should equal false
    scalarF32 v.Value |> should (equalWithin 1e-5f) 3.0f

// A11
[<Fact>]
let ``A11 scoped keeps None without error`` () =
    let mutable intermediate = Unchecked.defaultof<torch.Tensor>

    let r =
        scoped {
            let tmp = torch.ones ([| 2L |], dtype = torch.float32, device = torch.CPU)
            intermediate <- tmp
            return (None: Tensor option)
        }

    let v = r
    v |> should equal None
    intermediate.IsInvalid |> should equal true

// A12
[<Fact>]
let ``A12 scoped keeps custom DU with tensor`` () =
    let r =
        scoped {
            let t = torch.ones ([| 4L |], dtype = torch.float32, device = torch.CPU)
            return Logits t
        }

    match r with
    | Logits t ->
        t.IsInvalid |> should equal false
        scalarF32 t |> should (equalWithin 1e-5f) 4.0f
    | _ -> failwith "Expected Logits"

[<Fact>]
let ``A12b scoped keeps custom DU with multiple tensor fields`` () =
    let r =
        scoped {
            let k = torch.ones ([| 2L |], dtype = torch.float32, device = torch.CPU)
            let v = torch.zeros ([| 2L |], dtype = torch.float32, device = torch.CPU)
            return Features(k, v)
        }

    match r with
    | Features(k, v) ->
        k.IsInvalid |> should equal false
        v.IsInvalid |> should equal false
        scalarF32 k |> should (equalWithin 1e-5f) 2.0f
        scalarF32 v |> should (equalWithin 1e-5f) 0.0f
    | _ -> failwith "Expected Features"

// A13
[<Fact>]
let ``A13 scoped keeps record with option tensor field`` () =
    let r =
        scoped {
            let t = torch.ones ([| 2L |], dtype = torch.float32, device = torch.CPU)
            let m = torch.zeros ([| 2L |], dtype = torch.float32, device = torch.CPU)
            return { Value = t; Mask = Some m }
        }

    let out = r
    out.Value.IsInvalid |> should equal false
    out.Mask.IsSome |> should equal true
    out.Mask.Value.IsInvalid |> should equal false

[<Fact>]
let ``A13b scoped keeps record with None tensor field`` () =
    let r =
        scoped {
            let t = torch.ones ([| 2L |], dtype = torch.float32, device = torch.CPU)
            return { Value = t; Mask = None }
        }

    let out = r
    out.Value.IsInvalid |> should equal false
    out.Mask |> should equal None

// A14
[<Fact>]
let ``A14 scoped keeps list of tensor tuples`` () =
    let r =
        scoped {
            let a = torch.ones ([| 2L |], dtype = torch.float32, device = torch.CPU)
            let b = torch.zeros ([| 2L |], dtype = torch.float32, device = torch.CPU)
            let c = torch.ones ([| 2L |], dtype = torch.float32, device = torch.CPU)
            let d = torch.zeros ([| 2L |], dtype = torch.float32, device = torch.CPU)
            return [ (a, b); (c, d) ]
        }

    let pairs = r
    pairs |> List.length |> should equal 2

    for a, b in pairs do
        a.IsInvalid |> should equal false
        b.IsInvalid |> should equal false

// A15
[<Fact>]
let ``A15 scoped with non-tensor scalar return`` () =
    let mutable intermediate = Unchecked.defaultof<torch.Tensor>

    let r =
        scoped {
            let t = torch.ones ([| 2L |], dtype = torch.float32, device = torch.CPU)
            intermediate <- t
            let s = t.sum ()
            let v = s.ToSingle()
            return v
        }

    let v = r
    v |> should (equalWithin 1e-5f) 2.0f
    intermediate.IsInvalid |> should equal true

// ═══════════════════════════════════════════════════════════════
// B. Scope lifecycle: creation and disposal timing
// ═══════════════════════════════════════════════════════════════

// B1
[<Fact>]
let ``B1 scoped disposes intermediate tensors`` () =
    let mutable aInner = Unchecked.defaultof<torch.Tensor>
    let mutable bInner = Unchecked.defaultof<torch.Tensor>

    let t =
        scoped {
            let a = torch.ones ([| 2L; 3L |], dtype = torch.float32, device = torch.CPU)
            let b = torch.ones ([| 2L; 3L |], dtype = torch.float32, device = torch.CPU)
            aInner <- a
            bInner <- b
            let c = a.add b
            return c
        }


    t.IsInvalid |> should equal false
    scalarF32 t |> should (equalWithin 1e-5f) 12.0f
    aInner.IsInvalid |> should equal true
    bInner.IsInvalid |> should equal true

// B2
[<Fact>]
let ``B2 nested scoped disposes inner without affecting outer`` () =
    let mutable innerIntermediate = Unchecked.defaultof<torch.Tensor>

    let a, b =
        scoped {
            let a = torch.ones ([| 2L |], dtype = torch.float32, device = torch.CPU)

            let b =
                scoped {
                    let tmp = torch.ones ([| 2L |], dtype = torch.float32, device = torch.CPU)
                    innerIntermediate <- tmp
                    let doubled = tmp.add tmp
                    return doubled
                }

            return a, b
        }


    a.IsInvalid |> should equal false
    b.IsInvalid |> should equal false
    innerIntermediate.IsInvalid |> should equal true
    scalarF32 b |> should (equalWithin 1e-5f) 4.0f

// B3
[<Fact>]
let ``B3 error disposes all created tensors`` () =
    let mutable createdInner = Unchecked.defaultof<torch.Tensor>

    try
        scoped {
            let t = torch.ones ([| 2L |], dtype = torch.float32, device = torch.CPU)
            createdInner <- t
            failwith "fail"
        }
        |> ignore

        failwith "Expected exception"
    with _ ->
        ()

    createdInner.IsInvalid |> should equal true

// B4
[<Fact>]
let ``B4 loop with scoped disposes each iteration`` () =
    let intermediates = System.Collections.Generic.List<torch.Tensor>()

    for _ in 0..2 do
        scoped {
            let t = torch.ones ([| 3L |], dtype = torch.float32, device = torch.CPU)
            intermediates.Add(t)
            t.add t |> ignore
        }

    intermediates.Count |> should equal 3

    intermediates
    |> Seq.iter (fun t -> t.IsInvalid |> should equal true)

// B5
[<Fact>]
let ``B5 explicit keep and auto-keep on same tensor`` () =
    let r =
        scoped {
            let t = torch.ones ([| 2L |], dtype = torch.float32, device = torch.CPU)
            let _ = Tensor.keep t
            return t
        }

    let t = r
    t.IsInvalid |> should equal false
    scalarF32 t |> should (equalWithin 1e-5f) 2.0f

// B6
[<Fact>]
let ``B6 exception in scoped disposes tensors via scope`` () =
    let mutable createdInner = Unchecked.defaultof<torch.Tensor>

    let r =
        scoped {
            let t = torch.ones ([| 2L |], dtype = torch.float32, device = torch.CPU)
            createdInner <- t

            try
                failwith "unexpected"
                return ()
            with _ ->
                return ()
        }

    r |> ignore
    createdInner.IsInvalid |> should equal true

// B7
[<Fact>]
let ``B7 for loop inside scoped accumulates in scope`` () =
    let created = System.Collections.Generic.List<torch.Tensor>()

    let r =
        scoped {
            let mutable acc = 0.0f

            for _ in 0..4 do
                let t = torch.ones ([| 1L |], dtype = torch.float32, device = torch.CPU)
                created.Add(t)
                let s = t.ToSingle()
                acc <- acc + s

            return acc
        }

    let total = r
    total |> should (equalWithin 1e-5f) 5.0f
    created.Count |> should equal 5

    created
    |> Seq.iter (fun t -> t.IsInvalid |> should equal true)

// B8
[<Fact>]
let ``B8 use binding inside scoped disposes resource`` () =
    let mutable disposed = false

    let r =
        scoped {
            use _r =
                { new System.IDisposable with
                    member _.Dispose() = disposed <- true
                }

            let t = torch.ones ([| 2L |], dtype = torch.float32, device = torch.CPU)
            let _ = t.add t
            return ()
        }

    r |> ignore
    disposed |> should equal true

// ═══════════════════════════════════════════════════════════════
// C. Tensor.keep
// ═══════════════════════════════════════════════════════════════

// C1
[<Fact>]
let ``C1 keep moves tensor out of dispose scope`` () =
    use _scope = torch.NewDisposeScope()
    let t = torch.ones ([| 3L |], dtype = torch.float32, device = torch.CPU)
    let kept = Tensor.keep t
    kept.IsInvalid |> should equal false
    kept.shape |> should equal [| 3L |]

// C2
[<Fact>]
let ``C2 keep preserves tensor after scope exits`` () =
    let mutable keptRef = Unchecked.defaultof<Tensor>
    let mutable droppedInner = Unchecked.defaultof<torch.Tensor>

    do
        use _scope = torch.NewDisposeScope()
        let a = torch.ones ([| 2L |], dtype = torch.float32, device = torch.CPU)
        let b = torch.ones ([| 2L |], dtype = torch.float32, device = torch.CPU)
        droppedInner <- b
        keptRef <- Tensor.keep a

    keptRef.IsInvalid |> should equal false
    scalarF32 keptRef |> should (equalWithin 1e-5f) 2.0f
    droppedInner.IsInvalid |> should equal true

// C3
[<Fact>]
let ``C3 keep is idempotent`` () =
    use _scope = torch.NewDisposeScope()
    let t = torch.ones ([| 2L |], dtype = torch.float32, device = torch.CPU)
    let t1 = Tensor.keep t
    let t2 = Tensor.keep t1
    obj.ReferenceEquals(t, t2) |> should equal true
    t2.IsInvalid |> should equal false

// C4
[<Fact>]
let ``C4 keep without active scope is safe`` () =
    let t = torch.ones ([| 2L |], dtype = torch.float32, device = torch.CPU)
    let kept = Tensor.keep t
    kept.IsInvalid |> should equal false
    scalarF32 kept |> should (equalWithin 1e-5f) 2.0f

// ═══════════════════════════════════════════════════════════════
// D. Integration: realistic usage patterns
// ═══════════════════════════════════════════════════════════════

// D1
[<Fact>]
let ``D1 train step pattern: forward-loss-backward-step`` () =
    let intermediates = System.Collections.Generic.List<torch.Tensor>()

    let w = torch.randn ([| 3L; 1L |], dtype = torch.float32, device = torch.CPU)
    let w = w.requires_grad_ ()
    let x = torch.randn ([| 4L; 3L |], dtype = torch.float32, device = torch.CPU)
    let y = torch.randn ([| 4L; 1L |], dtype = torch.float32, device = torch.CPU)

    for _ in 1..3 do
        scoped {
            let pred = x.matmul w
            intermediates.Add(pred)
            let diff = pred.sub y
            let sq = diff.mul diff
            let loss = sq.mean ()
            loss.backward ()
        }

    intermediates.Count |> should equal 3

    intermediates
    |> Seq.iter (fun t -> t.IsInvalid |> should equal true)

// D2
[<Fact>]
let ``D2 noGrad combined with scoped`` () =
    let mutable intermediate = Unchecked.defaultof<torch.Tensor>

    let r =
        scoped {
            let w = torch.ones ([| 2L; 2L |], dtype = torch.float32, device = torch.CPU)
            let x = torch.ones ([| 2L; 2L |], dtype = torch.float32, device = torch.CPU)

            let pred =
                Toro.noGrad (fun () ->
                    scoped {
                        let tmp = x.add w
                        intermediate <- tmp
                        let result = tmp.matmul w
                        return result
                    })

            return pred
        }

    let pred = r
    pred.IsInvalid |> should equal false
    intermediate.IsInvalid |> should equal true

// ═══════════════════════════════════════════════════════════════
// E. scoped side-effect disposal (no return auto-keep)
// ═══════════════════════════════════════════════════════════════

// E1
[<Fact>]
let ``E1 scoped disposes intermediate tensors`` () =
    let mutable aInner = Unchecked.defaultof<torch.Tensor>

    scoped {
        let t = torch.ones ([| 3L |], dtype = torch.float32, device = torch.CPU)
        aInner <- t
        t.add t |> ignore
    }

    aInner.IsInvalid |> should equal true

[<Fact>]
let ``E2 scopedExplicit only preserves explicitly kept tensors`` () =
    let disposed =
        scopedExplicit {
            let tensor = torch.ones ([| 2L |], dtype = torch.float32, device = torch.CPU)
            return tensor
        }

    disposed.IsInvalid |> should equal true

    let kept =
        scopedExplicit {
            let tensor = torch.ones ([| 2L |], dtype = torch.float32, device = torch.CPU)
            return Tensor.keep tensor
        }

    kept.IsInvalid |> should equal false
    scalarF32 kept |> should (equalWithin 1e-5f) 2.0f

// E3
[<Fact>]
let ``E3 scoped with Tensor.keep preserves tensor`` () =
    let t =
        scoped {
            let t = torch.ones ([| 2L |], dtype = torch.float32, device = torch.CPU)
            return Tensor.keep t
        }

    t.IsInvalid |> should equal false
    scalarF32 t |> should (equalWithin 1e-5f) 2.0f

// E4
[<Fact>]
let ``E4 scoped in loop disposes each iteration`` () =
    let intermediates = System.Collections.Generic.List<torch.Tensor>()

    for _ in 0..2 do
        scoped {
            let t = torch.ones ([| 3L |], dtype = torch.float32, device = torch.CPU)
            intermediates.Add(t)
            t.add t |> ignore
        }

    intermediates.Count |> should equal 3

    intermediates
    |> Seq.iter (fun t -> t.IsInvalid |> should equal true)

// ═══════════════════════════════════════════════════════════════
// F. Edge cases
// ═══════════════════════════════════════════════════════════════

// F1
[<Fact>]
let ``F1 scoped keeps empty list`` () =
    let r =
        scoped {
            let _ = torch.ones ([| 1L |], dtype = torch.float32, device = torch.CPU)
            return ([]: Tensor list)
        }

    let ts = r
    ts |> should be Empty

// F2
[<Fact>]
let ``F2 scoped keeps empty array`` () =
    let r =
        scoped {
            let _ = torch.ones ([| 1L |], dtype = torch.float32, device = torch.CPU)
            return ([||]: Tensor array)
        }

    let ts = r
    ts |> should be Empty

// F3
[<Fact>]
let ``F3 error mid-computation disposes earlier tensors`` () =
    let mutable createdInner = Unchecked.defaultof<torch.Tensor>

    try
        scoped {
            let t = torch.ones ([| 2L; 3L |], dtype = torch.float32, device = torch.CPU)
            createdInner <- t
            t.reshape [| 7L; 7L |] |> ignore
        }
        |> ignore

        failwith "Expected exception"
    with _ ->
        ()

    createdInner.IsInvalid |> should equal true

// F4
[<Fact>]
let ``F4 keep inside scoped for side-effect retention`` () =
    let mutable retained = Unchecked.defaultof<Tensor>
    let mutable droppedInner = Unchecked.defaultof<torch.Tensor>

    let r =
        scoped {
            let a = torch.ones ([| 2L |], dtype = torch.float32, device = torch.CPU)
            let b = torch.ones ([| 2L |], dtype = torch.float32, device = torch.CPU)
            retained <- Tensor.keep a
            droppedInner <- b
        }

    r |> ignore
    retained.IsInvalid |> should equal false
    droppedInner.IsInvalid |> should equal true

// ═══════════════════════════════════════════════════════════════
// G. Scope safety: keepTensors only moves owned tensors
// ═══════════════════════════════════════════════════════════════

// G1
[<Fact>]
let ``G1 same tensor in two tuple slots does not double-move`` () =
    let r =
        scoped {
            let t = torch.ones ([| 2L |], dtype = torch.float32, device = torch.CPU)
            return t, t
        }

    let a, b = r
    obj.ReferenceEquals(a, b) |> should equal true
    a.IsInvalid |> should equal false
    scalarF32 a |> should (equalWithin 1e-5f) 2.0f

// G2
[<Fact>]
let ``G2 outer tensor returned from inner scoped stays in outer scope`` () =
    let r =
        scoped {
            let outer = torch.ones ([| 2L |], dtype = torch.float32, device = torch.CPU)

            let inner =
                scoped {
                    let tmp = torch.ones ([| 2L |], dtype = torch.float32, device = torch.CPU)
                    let sum = outer.add tmp
                    return sum
                }

            return outer, inner
        }

    let outer, inner = r
    outer.IsInvalid |> should equal false
    inner.IsInvalid |> should equal false
    scalarF32 outer |> should (equalWithin 1e-5f) 2.0f
    scalarF32 inner |> should (equalWithin 1e-5f) 4.0f

// G3
[<Fact>]
let ``G3 outer tensor returned from inner scoped is not moved past outer`` () =
    let mutable outerInner = Unchecked.defaultof<torch.Tensor>

    do
        use _outermost = torch.NewDisposeScope()

        let r =
            scoped {
                let outer = torch.ones ([| 2L |], dtype = torch.float32, device = torch.CPU)
                outerInner <- outer

                let passed = scoped { return outer }

                return passed
            }

        let t = r
        t.IsInvalid |> should equal false
        _outermost.Contains(outerInner) |> should equal true

    outerInner.IsInvalid |> should equal true

// G4
[<Fact>]
let ``G4 scoped with nested helper keeps result tensors`` () =
    let helperOp () : Tensor * Tensor =
        let a = torch.ones ([| 3L |], dtype = torch.float32, device = torch.CPU)
        let b = torch.zeros ([| 3L |], dtype = torch.float32, device = torch.CPU)
        a, b

    let r =
        scoped {
            let _ = torch.ones ([| 1L |], dtype = torch.float32, device = torch.CPU)
            return helperOp ()
        }

    let a, b = r
    a.IsInvalid |> should equal false
    b.IsInvalid |> should equal false
    scalarF32 a |> should (equalWithin 1e-5f) 3.0f
    scalarF32 b |> should (equalWithin 1e-5f) 0.0f

// G5
[<Fact>]
let ``G5 scoped keeps Map of tensors`` () =
    let mutable intermediateInner = Unchecked.defaultof<torch.Tensor>

    let r =
        scoped {
            let a = torch.ones ([| 2L |], dtype = torch.float32, device = torch.CPU)
            let b = torch.zeros ([| 2L |], dtype = torch.float32, device = torch.CPU)
            let tmp = torch.ones ([| 1L |], dtype = torch.float32, device = torch.CPU)
            intermediateInner <- tmp
            return Map [ "a", a; "b", b ]
        }

    let m = r
    m.Count |> should equal 2
    m["a"].IsInvalid |> should equal false
    m["b"].IsInvalid |> should equal false
    scalarF32 m["a"] |> should (equalWithin 1e-5f) 2.0f
    scalarF32 m["b"] |> should (equalWithin 1e-5f) 0.0f
    intermediateInner.IsInvalid |> should equal true

// G6
[<Fact>]
let ``G6 keep is idempotent within same scope`` () =
    use _scope = torch.NewDisposeScope()
    let t = torch.ones ([| 2L |], dtype = torch.float32, device = torch.CPU)
    let _ = Tensor.keep t
    let _ = Tensor.keep t
    t.IsInvalid |> should equal false
    scalarF32 t |> should (equalWithin 1e-5f) 2.0f

// G7
[<Fact>]
let ``G7 keep in nested scope does not move past outer`` () =
    let mutable keptRef = Unchecked.defaultof<Tensor>

    do
        use _outer = torch.NewDisposeScope()

        do
            use _inner = torch.NewDisposeScope()
            let t = torch.ones ([| 2L |], dtype = torch.float32, device = torch.CPU)
            keptRef <- Tensor.keep t

        keptRef.IsInvalid |> should equal false
        _outer.Contains(keptRef) |> should equal true

    keptRef.IsInvalid |> should equal true

// G8
[<Fact>]
let ``G8 double keep in nested scopes does not move two levels`` () =
    let mutable keptRef = Unchecked.defaultof<Tensor>

    do
        use _outermost = torch.NewDisposeScope()

        do
            use _outer = torch.NewDisposeScope()

            do
                use _inner = torch.NewDisposeScope()
                let t = torch.ones ([| 2L |], dtype = torch.float32, device = torch.CPU)
                keptRef <- Tensor.keep t
                let _ = Tensor.keep keptRef
                ()

            keptRef.IsInvalid |> should equal false
            _outer.Contains(keptRef) |> should equal true

        keptRef.IsInvalid |> should equal true
