open Toro
open Toro.NN

[<EntryPoint>]
let main _argv =
    result {
        let! x =
            Tensor.ofArray (
                array2D [|
                    [| 0f; 0f |] //
                    [| 0f; 1f |]
                    [| 1f; 0f |]
                    [| 1f; 1f |]
                |],
                Cpu
            )

        let! y = Tensor.ofArray (array2D [| [| 0f |]; [| 1f |]; [| 1f |]; [| 0f |] |], Cpu)

        let! l1 = Linear.init 2 16 F32 Cpu
        let! l2 = Linear.init 16 1 F32 Cpu

        let model =
            sequential {
                l1
                Relu
                l2
            }

        let! opt = AdamW.createWithLr 0.01 (Model.trainableVars model)

        printfn "Training XOR with AdamW..."

        for epoch in 1..500 do
            do!
                scoped {
                    opt.zeroGrad ()
                    let! pred = model.forward x
                    let! loss = Loss.mse pred y
                    do! loss.backward ()
                    do! opt.step ()

                    if epoch % 100 = 0 then
                        let! v = loss.toFloat32Scalar ()
                        printfn "  epoch %4d  loss = %.6f" epoch v
                }

        printfn ""
        printfn "Predictions (expected: 0, 1, 1, 0):"

        let! pred =
            Toro.noGrad (fun () ->
                result {
                    let! p = model.forward x
                    return! p.flattenAll ()
                })

        let labels = [|
            "0 XOR 0" //
            "0 XOR 1"
            "1 XOR 0"
            "1 XOR 1"
        |]

        for i in 0..3 do
            let! v = pred[i].toFloat32Scalar ()
            printfn "  %s = %.3f" labels[i] v

    }
    |> function
        | Ok() -> 0
        | Error e ->
            eprintfn "%A" e
            1
