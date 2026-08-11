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
            let! t = Tensor.randn ([ 4 ], F32, Cpu)
            return t
        }

    let t = unwrap r
    t.Shape |> should equal [ 4 ]
    t.Inner.IsInvalid |> should equal false

// A2
[<Fact>]
let ``A2 scoped with unit return`` () =
    let mutable flag = false

    let r =
        scoped {
            let! _ = Tensor.ones ([ 2 ], F32, Cpu)
            flag <- true
        }

    unwrap r |> ignore
    flag |> should equal true

// A3
[<Fact>]
let ``A3 scoped keeps 2-tuple of tensors`` () =
    let r =
        scoped {
            let! a = Tensor.zeros ([ 3 ], F32, Cpu)
            let! b = Tensor.ones ([ 3 ], F32, Cpu)
            return a, b
        }

    let a, b = unwrap r
    a.Inner.IsInvalid |> should equal false
    b.Inner.IsInvalid |> should equal false
    scalarF32 a |> should (equalWithin 1e-5f) 0.0f
    scalarF32 b |> should (equalWithin 1e-5f) 3.0f

// A4
[<Fact>]
let ``A4 scoped keeps 3-tuple of tensors`` () =
    let r =
        scoped {
            let! a = Tensor.zeros ([ 2 ], F32, Cpu)
            let! b = Tensor.ones ([ 2 ], F32, Cpu)
            let! c = a.add b
            return a, b, c
        }

    let a, b, c = unwrap r
    a.Inner.IsInvalid |> should equal false
    b.Inner.IsInvalid |> should equal false
    c.Inner.IsInvalid |> should equal false
    scalarF32 c |> should (equalWithin 1e-5f) 2.0f

// A5
[<Fact>]
let ``A5 scoped keeps mixed tuple`` () =
    let r =
        scoped {
            let! t = Tensor.ones ([ 3 ], F32, Cpu)
            return t, 42
        }

    let tensor, n = unwrap r
    tensor.Inner.IsInvalid |> should equal false
    scalarF32 tensor |> should (equalWithin 1e-5f) 3.0f
    n |> should equal 42

// A6
[<Fact>]
let ``A6 scoped keeps record`` () =
    let mutable intermediate = Unchecked.defaultof<torch.Tensor>

    let r =
        scoped {
            let! a = Tensor.ones ([ 2; 3 ], F32, Cpu)
            let! w = Tensor.zeros ([ 2; 2 ], F32, Cpu)
            let! tmp = Tensor.ones ([ 1 ], F32, Cpu)
            intermediate <- tmp.Inner
            return { Attn = a; Weights = w }
        }

    let out = unwrap r
    out.Attn.Inner.IsInvalid |> should equal false
    out.Weights.Inner.IsInvalid |> should equal false
    intermediate.IsInvalid |> should equal true

// A7
[<Fact>]
let ``A7 scoped keeps nested record`` () =
    let r =
        scoped {
            let! a = Tensor.ones ([ 2 ], F32, Cpu)
            let! w = Tensor.zeros ([ 2 ], F32, Cpu)
            let! l = Tensor.ones ([ 1 ], F32, Cpu)

            return {
                Output = { Attn = a; Weights = w }
                Loss = l
            }
        }

    let out = unwrap r
    out.Output.Attn.Inner.IsInvalid |> should equal false
    out.Output.Weights.Inner.IsInvalid |> should equal false
    out.Loss.Inner.IsInvalid |> should equal false

// A8
[<Fact>]
let ``A8 scoped keeps tensor list`` () =
    let r =
        scoped {
            let! a = Tensor.zeros ([ 2 ], F32, Cpu)
            let! b = Tensor.ones ([ 2 ], F32, Cpu)
            return [ a; b ]
        }

    let ts = unwrap r
    ts |> List.length |> should equal 2

    ts
    |> List.iter (fun t -> t.Inner.IsInvalid |> should equal false)

    scalarF32 ts[1] |> should (equalWithin 1e-5f) 2.0f

// A9
[<Fact>]
let ``A9 scoped keeps tensor array`` () =
    let r =
        scoped {
            let! a = Tensor.ones ([ 2 ], F32, Cpu)
            let! b = Tensor.ones ([ 2 ], F32, Cpu)
            return [| a; b |]
        }

    let ts = unwrap r
    ts |> Array.length |> should equal 2

    ts
    |> Array.iter (fun t -> t.Inner.IsInvalid |> should equal false)

// A10
[<Fact>]
let ``A10 scoped keeps Some tensor`` () =
    let r =
        scoped {
            let! t = Tensor.ones ([ 3 ], F32, Cpu)
            return Some t
        }

    let v = unwrap r
    v.IsSome |> should equal true
    v.Value.Inner.IsInvalid |> should equal false
    scalarF32 v.Value |> should (equalWithin 1e-5f) 3.0f

// A11
[<Fact>]
let ``A11 scoped keeps None without error`` () =
    let mutable intermediate = Unchecked.defaultof<torch.Tensor>

    let r =
        scoped {
            let! tmp = Tensor.ones ([ 2 ], F32, Cpu)
            intermediate <- tmp.Inner
            return (None: Tensor option)
        }

    let v = unwrap r
    v |> should equal None
    intermediate.IsInvalid |> should equal true

// A12
[<Fact>]
let ``A12 scoped keeps custom DU with tensor`` () =
    let r =
        scoped {
            let! t = Tensor.ones ([ 4 ], F32, Cpu)
            return Logits t
        }

    match unwrap r with
    | Logits t ->
        t.Inner.IsInvalid |> should equal false
        scalarF32 t |> should (equalWithin 1e-5f) 4.0f
    | _ -> failwith "Expected Logits"

[<Fact>]
let ``A12b scoped keeps custom DU with multiple tensor fields`` () =
    let r =
        scoped {
            let! k = Tensor.ones ([ 2 ], F32, Cpu)
            let! v = Tensor.zeros ([ 2 ], F32, Cpu)
            return Features(k, v)
        }

    match unwrap r with
    | Features(k, v) ->
        k.Inner.IsInvalid |> should equal false
        v.Inner.IsInvalid |> should equal false
        scalarF32 k |> should (equalWithin 1e-5f) 2.0f
        scalarF32 v |> should (equalWithin 1e-5f) 0.0f
    | _ -> failwith "Expected Features"

// A13
[<Fact>]
let ``A13 scoped keeps record with option tensor field`` () =
    let r =
        scoped {
            let! t = Tensor.ones ([ 2 ], F32, Cpu)
            let! m = Tensor.zeros ([ 2 ], F32, Cpu)
            return { Value = t; Mask = Some m }
        }

    let out = unwrap r
    out.Value.Inner.IsInvalid |> should equal false
    out.Mask.IsSome |> should equal true
    out.Mask.Value.Inner.IsInvalid |> should equal false

[<Fact>]
let ``A13b scoped keeps record with None tensor field`` () =
    let r =
        scoped {
            let! t = Tensor.ones ([ 2 ], F32, Cpu)
            return { Value = t; Mask = None }
        }

    let out = unwrap r
    out.Value.Inner.IsInvalid |> should equal false
    out.Mask |> should equal None

// A14
[<Fact>]
let ``A14 scoped keeps list of tensor tuples`` () =
    let r =
        scoped {
            let! a = Tensor.ones ([ 2 ], F32, Cpu)
            let! b = Tensor.zeros ([ 2 ], F32, Cpu)
            let! c = Tensor.ones ([ 2 ], F32, Cpu)
            let! d = Tensor.zeros ([ 2 ], F32, Cpu)
            return [ (a, b); (c, d) ]
        }

    let pairs = unwrap r
    pairs |> List.length |> should equal 2

    for a, b in pairs do
        a.Inner.IsInvalid |> should equal false
        b.Inner.IsInvalid |> should equal false

// A15
[<Fact>]
let ``A15 scoped with non-tensor scalar return`` () =
    let mutable intermediate = Unchecked.defaultof<torch.Tensor>

    let r =
        scoped {
            let! t = Tensor.ones ([ 2 ], F32, Cpu)
            intermediate <- t.Inner
            let! s = t.sumAll ()
            let! v = s.toFloat32Scalar ()
            return v
        }

    let v = unwrap r
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
            let! a = Tensor.ones ([ 2; 3 ], F32, Cpu)
            let! b = Tensor.ones ([ 2; 3 ], F32, Cpu)
            aInner <- a.Inner
            bInner <- b.Inner
            let! c = a.add b
            return c
        }
        |> unwrap

    t.Inner.IsInvalid |> should equal false
    scalarF32 t |> should (equalWithin 1e-5f) 12.0f
    aInner.IsInvalid |> should equal true
    bInner.IsInvalid |> should equal true

// B2
[<Fact>]
let ``B2 nested scoped disposes inner without affecting outer`` () =
    let mutable innerIntermediate = Unchecked.defaultof<torch.Tensor>

    let a, b =
        scoped {
            let! a = Tensor.ones ([ 2 ], F32, Cpu)

            let! b =
                scoped {
                    let! tmp = Tensor.ones ([ 2 ], F32, Cpu)
                    innerIntermediate <- tmp.Inner
                    let! doubled = tmp.add tmp
                    return doubled
                }

            return a, b
        }
        |> unwrap

    a.Inner.IsInvalid |> should equal false
    b.Inner.IsInvalid |> should equal false
    innerIntermediate.IsInvalid |> should equal true
    scalarF32 b |> should (equalWithin 1e-5f) 4.0f

// B3
[<Fact>]
let ``B3 error disposes all created tensors`` () =
    let mutable createdInner = Unchecked.defaultof<torch.Tensor>

    let r =
        scoped {
            let! t = Tensor.ones ([ 2 ], F32, Cpu)
            createdInner <- t.Inner
            return! Error(ToroError.Wrapped(System.InvalidOperationException "fail"))
        }

    r |> Result.isError |> should equal true
    createdInner.IsInvalid |> should equal true

// B4
[<Fact>]
let ``B4 loop with do! scoped disposes each iteration`` () =
    let intermediates = System.Collections.Generic.List<torch.Tensor>()

    let r =
        result {
            for _ in 0..2 do
                do!
                    scoped {
                        let! t = Tensor.ones ([ 3 ], F32, Cpu)
                        intermediates.Add(t.Inner)
                        let! _ = t.add t
                        return ()
                    }
        }

    unwrap r |> ignore
    intermediates.Count |> should equal 3

    intermediates
    |> Seq.iter (fun t -> t.IsInvalid |> should equal true)

// B5
[<Fact>]
let ``B5 explicit keep and auto-keep on same tensor`` () =
    let r =
        scoped {
            let! t = Tensor.ones ([ 2 ], F32, Cpu)
            let _ = Tensor.keep t
            return t
        }

    let t = unwrap r
    t.Inner.IsInvalid |> should equal false
    scalarF32 t |> should (equalWithin 1e-5f) 2.0f

// B6
[<Fact>]
let ``B6 exception in scoped disposes tensors via scope`` () =
    let mutable createdInner = Unchecked.defaultof<torch.Tensor>

    let r =
        scoped {
            let! t = Tensor.ones ([ 2 ], F32, Cpu)
            createdInner <- t.Inner

            try
                failwith "unexpected"
                return ()
            with _ ->
                return ()
        }

    unwrap r |> ignore
    createdInner.IsInvalid |> should equal true

// B7
[<Fact>]
let ``B7 for loop inside scoped accumulates in scope`` () =
    let created = System.Collections.Generic.List<torch.Tensor>()

    let r =
        scoped {
            let mutable acc = 0.0f

            for _ in 0..4 do
                let! t = Tensor.ones ([ 1 ], F32, Cpu)
                created.Add(t.Inner)
                let! s = t.toFloat32Scalar ()
                acc <- acc + s

            return acc
        }

    let total = unwrap r
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

            let! t = Tensor.ones ([ 2 ], F32, Cpu)
            let! _ = t.add t
            return ()
        }

    unwrap r |> ignore
    disposed |> should equal true

// ═══════════════════════════════════════════════════════════════
// C. Tensor.keep
// ═══════════════════════════════════════════════════════════════

// C1
[<Fact>]
let ``C1 keep moves tensor out of dispose scope`` () =
    use _scope = torch.NewDisposeScope()
    let t = Tensor.ones ([ 3 ], F32, Cpu) |> unwrap
    let kept = Tensor.keep t
    kept.Inner.IsInvalid |> should equal false
    kept.Shape |> should equal [ 3 ]

// C2
[<Fact>]
let ``C2 keep preserves tensor after scope exits`` () =
    let mutable keptRef = Unchecked.defaultof<Tensor>
    let mutable droppedInner = Unchecked.defaultof<torch.Tensor>

    do
        use _scope = torch.NewDisposeScope()
        let a = Tensor.ones ([ 2 ], F32, Cpu) |> unwrap
        let b = Tensor.ones ([ 2 ], F32, Cpu) |> unwrap
        droppedInner <- b.Inner
        keptRef <- Tensor.keep a

    keptRef.Inner.IsInvalid |> should equal false
    scalarF32 keptRef |> should (equalWithin 1e-5f) 2.0f
    droppedInner.IsInvalid |> should equal true

// C3
[<Fact>]
let ``C3 keep is idempotent`` () =
    use _scope = torch.NewDisposeScope()
    let t = Tensor.ones ([ 2 ], F32, Cpu) |> unwrap
    let t1 = Tensor.keep t
    let t2 = Tensor.keep t1
    obj.ReferenceEquals(t, t2) |> should equal true
    t2.Inner.IsInvalid |> should equal false

// C4
[<Fact>]
let ``C4 keep without active scope is safe`` () =
    let t = Tensor.ones ([ 2 ], F32, Cpu) |> unwrap
    let kept = Tensor.keep t
    kept.Inner.IsInvalid |> should equal false
    scalarF32 kept |> should (equalWithin 1e-5f) 2.0f

// ═══════════════════════════════════════════════════════════════
// D. Integration: realistic usage patterns
// ═══════════════════════════════════════════════════════════════

// D1
[<Fact>]
let ``D1 train step pattern: forward-loss-backward-step`` () =
    let intermediates = System.Collections.Generic.List<torch.Tensor>()

    let r =
        result {
            let! w = Tensor.randn ([ 3; 1 ], F32, Cpu)
            let! w = w.requiresGrad ()
            let! x = Tensor.randn ([ 4; 3 ], F32, Cpu)
            let! y = Tensor.randn ([ 4; 1 ], F32, Cpu)

            for _ in 1..3 do
                do!
                    scoped {
                        let! pred = x.matmul w
                        intermediates.Add(pred.Inner)
                        let! diff = pred.sub y
                        let! sq = diff.mul diff
                        let! loss = sq.meanAll ()
                        do! loss.backward ()
                        return ()
                    }
        }

    unwrap r |> ignore
    intermediates.Count |> should equal 3

    intermediates
    |> Seq.iter (fun t -> t.IsInvalid |> should equal true)

// D2
[<Fact>]
let ``D2 noGrad combined with scoped`` () =
    let mutable intermediate = Unchecked.defaultof<torch.Tensor>

    let r =
        scoped {
            let! w = Tensor.ones ([ 2; 2 ], F32, Cpu)
            let! x = Tensor.ones ([ 2; 2 ], F32, Cpu)

            let! pred =
                Toro.noGrad (fun () ->
                    scoped {
                        let! tmp = x.add w
                        intermediate <- tmp.Inner
                        let! result = tmp.matmul w
                        return result
                    })

            return pred
        }

    let pred = unwrap r
    pred.Inner.IsInvalid |> should equal false
    intermediate.IsInvalid |> should equal true

// ═══════════════════════════════════════════════════════════════
// E. disposeScope
// ═══════════════════════════════════════════════════════════════

// E1
[<Fact>]
let ``E1 disposeScope disposes intermediate tensors`` () =
    let mutable aInner = Unchecked.defaultof<torch.Tensor>

    let r =
        result {
            use! _s = disposeScope ()
            let! t = Tensor.ones ([ 3 ], F32, Cpu)
            aInner <- t.Inner
            let! _ = t.add t
            return ()
        }

    unwrap r |> ignore
    aInner.IsInvalid |> should equal true

// E2
[<Fact>]
let ``E2 disposeScope does NOT auto-keep return tensors`` () =
    let mutable tInner = Unchecked.defaultof<torch.Tensor>

    let r =
        result {
            use! _s = disposeScope ()
            let! t = Tensor.ones ([ 2 ], F32, Cpu)
            tInner <- t.Inner
        }

    unwrap r |> ignore
    tInner.IsInvalid |> should equal true

// E3
[<Fact>]
let ``E3 disposeScope with Tensor.keep preserves tensor`` () =
    let r =
        result {
            use! _s = disposeScope ()
            let! t = Tensor.ones ([ 2 ], F32, Cpu)
            let kept = Tensor.keep t
            return kept
        }

    let t = unwrap r
    t.Inner.IsInvalid |> should equal false
    scalarF32 t |> should (equalWithin 1e-5f) 2.0f

// E4
[<Fact>]
let ``E4 disposeScope in loop disposes each iteration`` () =
    let intermediates = System.Collections.Generic.List<torch.Tensor>()

    let r =
        result {
            for _ in 0..2 do
                use! _s = disposeScope ()
                let! t = Tensor.ones ([ 3 ], F32, Cpu)
                intermediates.Add(t.Inner)
                let! _ = t.add t
                ()
        }

    unwrap r |> ignore
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
            let! _ = Tensor.ones ([ 1 ], F32, Cpu)
            return ([]: Tensor list)
        }

    let ts = unwrap r
    ts |> should be Empty

// F2
[<Fact>]
let ``F2 scoped keeps empty array`` () =
    let r =
        scoped {
            let! _ = Tensor.ones ([ 1 ], F32, Cpu)
            return ([||]: Tensor array)
        }

    let ts = unwrap r
    ts |> should be Empty

// F3
[<Fact>]
let ``F3 let! error mid-computation disposes earlier tensors`` () =
    let mutable createdInner = Unchecked.defaultof<torch.Tensor>

    let r =
        scoped {
            let! t = Tensor.ones ([ 2; 3 ], F32, Cpu)
            createdInner <- t.Inner
            let! _bad = t.reshape [ 7; 7 ]
            return ()
        }

    r |> Result.isError |> should equal true
    createdInner.IsInvalid |> should equal true

// F4
[<Fact>]
let ``F4 keep inside scoped for side-effect retention`` () =
    let mutable retained = Unchecked.defaultof<Tensor>
    let mutable droppedInner = Unchecked.defaultof<torch.Tensor>

    let r =
        scoped {
            let! a = Tensor.ones ([ 2 ], F32, Cpu)
            let! b = Tensor.ones ([ 2 ], F32, Cpu)
            retained <- Tensor.keep a
            droppedInner <- b.Inner
        }

    unwrap r |> ignore
    retained.Inner.IsInvalid |> should equal false
    droppedInner.IsInvalid |> should equal true

// ═══════════════════════════════════════════════════════════════
// G. Scope safety: keepTensors only moves owned tensors
// ═══════════════════════════════════════════════════════════════

// G1
[<Fact>]
let ``G1 same tensor in two tuple slots does not double-move`` () =
    let r =
        scoped {
            let! t = Tensor.ones ([ 2 ], F32, Cpu)
            return t, t
        }

    let a, b = unwrap r
    obj.ReferenceEquals(a, b) |> should equal true
    a.Inner.IsInvalid |> should equal false
    scalarF32 a |> should (equalWithin 1e-5f) 2.0f

// G2
[<Fact>]
let ``G2 outer tensor returned from inner scoped stays in outer scope`` () =
    let r =
        scoped {
            let! outer = Tensor.ones ([ 2 ], F32, Cpu)

            let! inner =
                scoped {
                    let! tmp = Tensor.ones ([ 2 ], F32, Cpu)
                    let! sum = outer.add tmp
                    return sum
                }

            return outer, inner
        }

    let outer, inner = unwrap r
    outer.Inner.IsInvalid |> should equal false
    inner.Inner.IsInvalid |> should equal false
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
                let! outer = Tensor.ones ([ 2 ], F32, Cpu)
                outerInner <- outer.Inner

                let! passed = scoped { return outer }

                return passed
            }

        let t = unwrap r
        t.Inner.IsInvalid |> should equal false
        _outermost.Contains(outerInner) |> should equal true

    outerInner.IsInvalid |> should equal true

// G4
[<Fact>]
let ``G4 scoped with return! keeps result tensors`` () =
    let helperOp () : Result<Tensor * Tensor, ToroError> =
        result {
            let! a = Tensor.ones ([ 3 ], F32, Cpu)
            let! b = Tensor.zeros ([ 3 ], F32, Cpu)
            return a, b
        }

    let r =
        scoped {
            let! _ = Tensor.ones ([ 1 ], F32, Cpu)
            return! helperOp ()
        }

    let a, b = unwrap r
    a.Inner.IsInvalid |> should equal false
    b.Inner.IsInvalid |> should equal false
    scalarF32 a |> should (equalWithin 1e-5f) 3.0f
    scalarF32 b |> should (equalWithin 1e-5f) 0.0f

// G5
[<Fact>]
let ``G5 scoped keeps Map of tensors`` () =
    let mutable intermediateInner = Unchecked.defaultof<torch.Tensor>

    let r =
        scoped {
            let! a = Tensor.ones ([ 2 ], F32, Cpu)
            let! b = Tensor.zeros ([ 2 ], F32, Cpu)
            let! tmp = Tensor.ones ([ 1 ], F32, Cpu)
            intermediateInner <- tmp.Inner
            return Map [ "a", a; "b", b ]
        }

    let m = unwrap r
    m.Count |> should equal 2
    m["a"].Inner.IsInvalid |> should equal false
    m["b"].Inner.IsInvalid |> should equal false
    scalarF32 m["a"] |> should (equalWithin 1e-5f) 2.0f
    scalarF32 m["b"] |> should (equalWithin 1e-5f) 0.0f
    intermediateInner.IsInvalid |> should equal true

// G6
[<Fact>]
let ``G6 keep is idempotent within same scope`` () =
    use _scope = torch.NewDisposeScope()
    let t = Tensor.ones ([ 2 ], F32, Cpu) |> unwrap
    let _ = Tensor.keep t
    let _ = Tensor.keep t
    t.Inner.IsInvalid |> should equal false
    scalarF32 t |> should (equalWithin 1e-5f) 2.0f

// G7
[<Fact>]
let ``G7 keep in nested scope does not move past outer`` () =
    let mutable keptRef = Unchecked.defaultof<Tensor>

    do
        use _outer = torch.NewDisposeScope()

        do
            use _inner = torch.NewDisposeScope()
            let t = Tensor.ones ([ 2 ], F32, Cpu) |> unwrap
            keptRef <- Tensor.keep t

        keptRef.Inner.IsInvalid |> should equal false
        _outer.Contains(keptRef.Inner) |> should equal true

    keptRef.Inner.IsInvalid |> should equal true

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
                let t = Tensor.ones ([ 2 ], F32, Cpu) |> unwrap
                keptRef <- Tensor.keep t
                let _ = Tensor.keep keptRef
                ()

            keptRef.Inner.IsInvalid |> should equal false
            _outer.Contains(keptRef.Inner) |> should equal true

        keptRef.Inner.IsInvalid |> should equal true
