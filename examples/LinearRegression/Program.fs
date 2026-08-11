open Toro

// y = 3x + 2 by gradient descent

[<EntryPoint>]
let main _argv =
    // --- data generation ---
    let x = Tensor.randn ([ 100; 1 ], F32, Cpu)
    let noise = Tensor.randn ([ 100; 1 ], F32, Cpu)
    let noise = noise * 0.1
    let y = x * 3.0 + 2.0 + noise

    // --- parameter initialization ---
    let w = Tensor.randn ([ 1; 1 ], F32, Cpu)
    let w = w.requiresGrad ()
    let b = Tensor.zeros ([ 1 ], F32, Cpu)
    let b = b.requiresGrad ()

    let lr = 0.1

    printfn "Linear regression: learning y = 3x + 2"
    printfn "lr = %.1f, steps = 200" lr
    printfn ""

    for step in 1..200 do
        // forward: pred = x @ w + b
        let pred = x.matmul w
        let pred = pred + b

        // loss = mean((pred - y)^2)
        let diff = pred - y
        let loss = (diff * diff).meanAll ()

        // backward
        loss.backward ()

        // SGD step (no_grad scope via copyInPlace)
        let gw = w.grad ()
        w.copyInPlace (w - gw * lr)
        w.zeroGrad ()

        let gb = b.grad ()
        b.copyInPlace (b - gb * lr)
        b.zeroGrad ()

        if step % 50 = 0 || step = 1 then
            let wVal = w.toFloat32Scalar ()
            let bVal = b.toFloat32Scalar ()
            printfn "  step %3d  loss=%.6f  w=%.4f  b=%.4f" step (loss.item ()) wVal bVal

    printfn ""

    let wFinal = w.toFloat32Scalar ()
    let bFinal = b.toFloat32Scalar ()
    printfn "Learned:  y = %.4f * x + %.4f" wFinal bFinal
    printfn "Expected: y = 3.0000 * x + 2.0000"
    0
