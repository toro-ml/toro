// Simple GCN node classification on a synthetic graph.
// 8 nodes, 2 classes. Train on 4 labeled nodes, evaluate on remaining 4.
open TorchSharp
open Toro
open Toro.NN
open Toro.GNN

/// Build a small graph: two clusters connected by a bridge edge.
/// Cluster A (nodes 0-3): class 0
/// Cluster B (nodes 4-7): class 1
let buildGraph () =
    let edgesSource = [| 0L; 1L; 0L; 2L; 1L; 2L; 1L; 3L; 4L; 5L; 4L; 6L; 5L; 6L; 5L; 7L; 3L; 4L |]
    let edgesTarget = [| 1L; 0L; 2L; 0L; 2L; 1L; 3L; 1L; 5L; 4L; 6L; 4L; 6L; 5L; 7L; 5L; 4L; 3L |]

    let edgeIndex =
        torch.tensor (Array.append edgesSource edgesTarget, device = torch.CPU)
        |> fun t -> t.reshape ([| 2L; int64 edgesSource.Length |])

    let x = torch.randn ([| 8L; 4L |], dtype = torch.float32, device = torch.CPU)
    (x, edgeIndex)

type GcnModel = { Conv1: GCNConv; Conv2: GCNConv }

let initModel () =
    let conv1 = GCNConv.init 4 16 torch.float32 torch.CPU
    let conv2 = GCNConv.init 16 2 torch.float32 torch.CPU
    { Conv1 = conv1; Conv2 = conv2 }

let forward (model: GcnModel) (x: Tensor) (edgeIndex: Tensor) =
    let h = model.Conv1.forward (x, edgeIndex)
    let h = h.relu ()
    model.Conv2.forward (h, edgeIndex)

[<EntryPoint>]
let main _argv =
    let (x, edgeIndex) = buildGraph ()

    // Labels: nodes 0-3 -> class 0, nodes 4-7 -> class 1
    let labels =
        torch.tensor ([| 0L; 0L; 0L; 0L; 1L; 1L; 1L; 1L |], dtype = torch.int64, device = torch.CPU)

    // Train mask: nodes 0, 1, 4, 5
    let trainIdx = [| 0; 1; 4; 5 |]
    // Test mask: nodes 2, 3, 6, 7
    let testIdx = [| 2; 3; 6; 7 |]

    let model = initModel ()

    let opt =
        AdamW.createWithLr 0.01 (model |> Model.state |> ModelState.trainableParams)

    printfn "Training 2-layer GCN on synthetic graph (8 nodes, 2 classes)..."
    printfn ""

    for epoch in 1..200 do
        scoped {
            let logits = forward model x edgeIndex

            let trainIdxTensor = torch.tensor (trainIdx |> Array.map int64, dtype = torch.int64)

            let trainLogits = logits.index (torch.TensorIndex.Tensor trainIdxTensor)

            let trainLabels = labels.index (torch.TensorIndex.Tensor trainIdxTensor)

            opt.zeroGrad ()
            let loss = Loss.crossEntropy trainLogits trainLabels
            loss.backward ()
            opt.step ()

            if epoch % 50 = 0 then
                let v = loss.ToSingle()
                printfn "  epoch %4d  loss = %.4f" epoch v
        }

    // Evaluate
    printfn ""
    printfn "Evaluation on test nodes:"

    let logits = Toro.noGrad (fun () -> forward model x edgeIndex)

    let preds = logits.argmax (int64 1)

    for i in testIdx do
        let pred = preds[i].ToSingle()
        let label = labels[i].ToSingle()
        let correct = if int pred = int label then "OK" else "MISS"
        printfn "  node %d: pred=%d label=%d  %s" i (int pred) (int label) correct

    0
