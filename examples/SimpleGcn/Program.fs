/// Simple GCN node classification on a synthetic graph.
/// 8 nodes, 2 classes. Train on 4 labeled nodes, evaluate on remaining 4.
open Toro
open Toro.NN
open Toro.GNN

/// Build a small graph: two clusters connected by a bridge edge.
/// Cluster A (nodes 0-3): class 0
/// Cluster B (nodes 4-7): class 1
let buildGraph () =
    result {
        let edgesSource = [| 0L; 1L; 0L; 2L; 1L; 2L; 1L; 3L; 4L; 5L; 4L; 6L; 5L; 6L; 5L; 7L; 3L; 4L |]
        let edgesTarget = [| 1L; 0L; 2L; 0L; 2L; 1L; 3L; 1L; 5L; 4L; 6L; 4L; 6L; 5L; 7L; 5L; 4L; 3L |]

        let! edgeIndex =
            Tensor.ofTorchTensor (
                TorchSharp.torch
                    .tensor(Array.append edgesSource edgesTarget, dtype = TorchSharp.torch.int64)
                    .reshape ([| 2L; int64 edgesSource.Length |])
            )

        let! x = Tensor.randn ([ 8; 4 ], F32, Cpu)
        return (x, edgeIndex)
    }

type GcnModel = { Conv1: GCNConv; Conv2: GCNConv }

let initModel () =
    result {
        let! conv1 = GCNConv.init 4 16 F32 Cpu
        let! conv2 = GCNConv.init 16 2 F32 Cpu
        return { Conv1 = conv1; Conv2 = conv2 }
    }

let forward (model: GcnModel) (x: Tensor) (edgeIndex: Tensor) =
    result {
        let! h = model.Conv1.forward (x, edgeIndex)
        let! h = h.relu ()
        return! model.Conv2.forward (h, edgeIndex)
    }

[<EntryPoint>]
let main _argv =
    result {
        let! (x, edgeIndex) = buildGraph ()

        // Labels: nodes 0-3 -> class 0, nodes 4-7 -> class 1
        let! labels = Tensor.ofTorchTensor (TorchSharp.torch.tensor ([| 0L; 0L; 0L; 0L; 1L; 1L; 1L; 1L |]: int64 array))

        // Train mask: nodes 0, 1, 4, 5
        let trainIdx = [| 0; 1; 4; 5 |]
        // Test mask: nodes 2, 3, 6, 7
        let testIdx = [| 2; 3; 6; 7 |]

        let! model = initModel ()
        let! opt = AdamW.createWithLr 0.01 (Model.trainableVars model)
        let opt = opt :> IOptimizer

        printfn "Training 2-layer GCN on synthetic graph (8 nodes, 2 classes)..."
        printfn ""

        for epoch in 1..200 do
            let! logits = forward model x edgeIndex

            // Select train nodes and compute cross-entropy loss
            let! trainLogits =
                Tensor.ofTorchTensor (
                    logits.Inner.index (
                        TorchSharp.torch.TensorIndex.Tensor(
                            TorchSharp.torch.tensor (trainIdx |> Array.map int64, dtype = TorchSharp.torch.int64)
                        )
                    )
                )

            let! trainLabels =
                Tensor.ofTorchTensor (
                    labels.Inner.index (
                        TorchSharp.torch.TensorIndex.Tensor(
                            TorchSharp.torch.tensor (trainIdx |> Array.map int64, dtype = TorchSharp.torch.int64)
                        )
                    )
                )

            let! loss = Loss.crossEntropy trainLogits trainLabels
            do! opt.backwardStep loss

            if epoch % 50 = 0 then
                let! v = loss.toFloat32Scalar ()
                printfn "  epoch %4d  loss = %.4f" epoch v

        // Evaluate
        printfn ""
        printfn "Evaluation on test nodes:"

        let! logits = Toro.noGrad (fun () -> forward model x edgeIndex)

        let! preds = logits.argmax (1)

        for i in testIdx do
            let! pred = preds[i].toFloat32Scalar ()
            let! label = labels[i].toFloat32Scalar ()
            let correct = if int pred = int label then "OK" else "MISS"
            printfn "  node %d: pred=%d label=%d  %s" i (int pred) (int label) correct
    }
    |> function
        | Ok() -> 0
        | Error e ->
            eprintfn "%A" e
            1
