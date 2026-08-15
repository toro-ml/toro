open TorchSharp
open Toro
open Toro.NN

[<EntryPoint>]
let main _argv =
    let x =
        torch.tensor (
            array2D [|
                [| 0f; 0f |] //
                [| 0f; 1f |]
                [| 1f; 0f |]
                [| 1f; 1f |]
            |],
            device = torch.CPU
        )

    let y =
        torch.tensor (array2D [| [| 0f |]; [| 1f |]; [| 1f |]; [| 0f |] |], device = torch.CPU)

    let l1 = Linear.init 2 16 torch.float32 torch.CPU
    let l2 = Linear.init 16 1 torch.float32 torch.CPU

    let model =
        sequential {
            l1
            Relu
            l2
        }

    let opt =
        AdamW.createWithLr 0.01 (model |> Model.state |> ModelState.trainableParams)

    printfn "Training XOR with AdamW..."

    for epoch in 1..500 do
        scoped {
            opt.zeroGrad ()
            let pred = model.forward x
            let loss = Loss.mse pred y
            loss.backward ()
            opt.step ()

            if epoch % 100 = 0 then
                let v = loss.ToSingle()
                printfn "  epoch %4d  loss = %.6f" epoch v
        }

    printfn ""
    printfn "Predictions (expected: 0, 1, 1, 0):"

    let pred =
        Toro.noGrad (fun () ->
            let p = model.forward x
            p.flatten ())

    let labels = [|
        "0 XOR 0" //
        "0 XOR 1"
        "1 XOR 0"
        "1 XOR 1"
    |]

    for i in 0..3 do
        let v = pred[i].ToSingle()
        printfn "  %s = %.3f" labels[i] v

    0
