open Toro
open Toro.NN

let unwrap r =
    match r with
    | Ok v -> v
    | Error e -> failwithf "%A" e

[<EntryPoint>]
let main _argv =
    // XOR dataset
    let x =
        Tensor.ofFloat32Array2D (
            [| [| 0f; 0f |]; [| 0f; 1f |]; [| 1f; 0f |]; [| 1f; 1f |] |],
            Cpu
        )
        |> unwrap

    let y =
        Tensor.ofFloat32Array2D ([| [| 0f |]; [| 1f |]; [| 1f |]; [| 0f |] |], Cpu)
        |> unwrap

    let varmap = VarMap()
    let vb = VarBuilder.fromVarMap varmap F32 Cpu

    let model =
        result {
            let! l1 = Linear.create 2 16 (vb |> VarBuilder.pp "l1")
            let! l2 = Linear.create 16 1 (vb |> VarBuilder.pp "l2")

            return Sequential.create [ l1 :> IModule; Relu :> IModule; l2 :> IModule ]
        }
        |> unwrap

    let opt = AdamW.createWithLr 0.01 (varmap.allVars ()) |> unwrap :> IOptimizer

    printfn "Training XOR with AdamW..."

    for epoch in 1..500 do
        let loss =
            result {
                let! pred = model.forward x
                return! Loss.mse pred y
            }
            |> unwrap

        opt.backwardStep loss |> unwrap

        if epoch % 100 = 0 then
            let v = loss.toFloat32Scalar () |> unwrap
            printfn "  epoch %4d  loss = %.6f" epoch v

    printfn ""
    printfn "Predictions (expected: 0, 1, 1, 0):"

    let pred =
        Toro.noGrad (fun () ->
            let p = model.forward x |> unwrap
            p.flattenAll () |> unwrap)

    let labels = [| "0 XOR 0"; "0 XOR 1"; "1 XOR 0"; "1 XOR 1" |]

    let idx = Tensor.arange (4.0, I64, Cpu) |> unwrap

    for i in 0..3 do
        let idxI = Tensor.full ([ 1 ], float i, I64, Cpu) |> unwrap
        let scalar = pred.indexSelect (0, idxI) |> unwrap
        let v = scalar.toFloat32Scalar () |> unwrap
        printfn "  %s = %.3f" labels[i] v

    0
