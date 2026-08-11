open TorchSharp
open Toro

// y = 3x + 2 by gradient descent

[<EntryPoint>]
let main _argv =
    // --- data generation ---
    let x = torch.randn ([| 100L; 1L |], dtype = torch.float32, device = torch.CPU)
    let noise = torch.randn ([| 100L; 1L |], dtype = torch.float32, device = torch.CPU)
    let noise = noise * 0.1
    let y = x * 3.0 + 2.0 + noise

    // --- parameter initialization ---
    let w = torch.randn ([| 1L; 1L |], dtype = torch.float32, device = torch.CPU)
    let w = w.requires_grad_ ()
    let b = torch.zeros ([| 1L |], dtype = torch.float32, device = torch.CPU)
    let b = b.requires_grad_ ()

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
        let loss = (diff * diff).mean ()

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
            let wVal = w.ToSingle()
            let bVal = b.ToSingle()
            printfn "  step %3d  loss=%.6f  w=%.4f  b=%.4f" step (loss.ToDouble()) wVal bVal

    printfn ""

    let wFinal = w.ToSingle()
    let bFinal = b.ToSingle()
    printfn "Learned:  y = %.4f * x + %.4f" wFinal bFinal
    printfn "Expected: y = 3.0000 * x + 2.0000"
    0
