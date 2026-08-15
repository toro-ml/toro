open System
open System.IO
open System.Net.Http
open System.Text.Json
open TorchSharp
open Toro
open Toro.Hub
open Toro.NN
open Toro.Vision

let repoId = "microsoft/resnet-18"
let revision = "65a5785d9156231087c481e0c7dd33a5ff6f7e3e"

type ConvNorm = { Conv: Conv2d; Norm: BatchNorm }

type BasicBlock = {
    Layers: ConvNorm list
    Shortcut: ConvNorm option
}

type ResNet18 = {
    Stem: ConvNorm
    Stages: BasicBlock list list
    Classifier: Linear
}

let convNorm inChannels outChannels kernelSize stride padding =
    let config = {
        Conv2dConfig.defaultConfig with
            Stride = stride
            Padding = padding
    }

    {
        Conv = Conv2d.initNoBias inChannels outChannels kernelSize config torch.float32 torch.CPU
        Norm = BatchNorm.initDefault outChannels torch.float32 torch.CPU
    }

let createBlock inChannels outChannels stride = {
    Layers = [
        convNorm inChannels outChannels 3L stride 1L
        convNorm outChannels outChannels 3L 1L 1L
    ]
    Shortcut =
        if stride <> 1L || inChannels <> outChannels then
            Some(convNorm inChannels outChannels 1L stride 0L)
        else
            None
}

let createStage inChannels outChannels stride = [
    createBlock inChannels outChannels stride
    createBlock outChannels outChannels 1L
]

let createModel () = {
    Stem = convNorm 3L 64L 7L 2L 3L
    Stages = [
        createStage 64L 64L 1L
        createStage 64L 128L 2L
        createStage 128L 256L 2L
        createStage 256L 512L 2L
    ]
    Classifier = Linear.init 512 1000 torch.float32 torch.CPU
}

let forwardConvNorm activation (layer: ConvNorm) input =
    let output = layer.Conv.forward input |> layer.Norm.forwardT false
    if activation then output.relu () else output

let forwardBlock (block: BasicBlock) input =
    let residual =
        block.Shortcut
        |> Option.map (fun shortcut -> forwardConvNorm false shortcut input)
        |> Option.defaultValue input

    let hidden =
        input
        |> forwardConvNorm true block.Layers[0]
        |> forwardConvNorm false block.Layers[1]

    (hidden + residual).relu ()

let forward (model: ResNet18) input =
    let hidden = forwardConvNorm true model.Stem input
    let hidden = torch.nn.functional.max_pool2d (hidden, 3L, stride = 2L, padding = 1L)

    let hidden =
        model.Stages
        |> List.collect id
        |> List.fold (fun state block -> forwardBlock block state) hidden

    let hidden = torch.nn.functional.adaptive_avg_pool2d (hidden, 1L)
    hidden.flatten (1L, -1L) |> model.Classifier.forward

let nameMapping =
    let norm sourcePrefix targetPrefix = [
        NameRule.rewrite (sourcePrefix + ".weight") (targetPrefix + ".Weight")
        NameRule.rewrite (sourcePrefix + ".bias") (targetPrefix + ".Bias")
        NameRule.rewrite (sourcePrefix + ".running_mean") (targetPrefix + ".RunningMean")
        NameRule.rewrite (sourcePrefix + ".running_var") (targetPrefix + ".RunningVar")
    ]

    let layerSource = "resnet.encoder.stages.{stage}.layers.{block}.layer.{layer}"
    let layerTarget = "Stages.{stage}.{block}.Layers.{layer}"
    let shortcutSource = "resnet.encoder.stages.{stage}.layers.{block}.shortcut"
    let shortcutTarget = "Stages.{stage}.{block}.Shortcut"

    NameMapping.create [
        NameRule.ignoreSuffix "num_batches_tracked"
        NameRule.rename "resnet.embedder.embedder.convolution.weight" "Stem.Conv.Weight"
        NameRule.rename "resnet.embedder.embedder.normalization.weight" "Stem.Norm.Weight"
        NameRule.rename "resnet.embedder.embedder.normalization.bias" "Stem.Norm.Bias"
        NameRule.rename "resnet.embedder.embedder.normalization.running_mean" "Stem.Norm.RunningMean"
        NameRule.rename "resnet.embedder.embedder.normalization.running_var" "Stem.Norm.RunningVar"
        NameRule.rewrite (layerSource + ".convolution.weight") (layerTarget + ".Conv.Weight")
        yield! norm (layerSource + ".normalization") (layerTarget + ".Norm")
        NameRule.rewrite (shortcutSource + ".convolution.weight") (shortcutTarget + ".Conv.Weight")
        yield! norm (shortcutSource + ".normalization") (shortcutTarget + ".Norm")
        NameRule.rename "classifier.1.weight" "Classifier.Weight"
        NameRule.rename "classifier.1.bias" "Classifier.Bias"
    ]

let hubFile filename = {
    RepoId = repoId
    Revision = revision
    Filename = filename
}

let loadJson filename =
    let path = Hub.download (hubFile filename) |> Async.RunSynchronously
    JsonDocument.Parse(File.ReadAllText path)

let loadPreprocessorConfig () =
    use document = loadJson "preprocessor_config.json"
    let root = document.RootElement
    let size = root.GetProperty("size").GetInt32()
    let cropPct = root.GetProperty("crop_pct").GetDouble()

    let values (name: string) =
        root.GetProperty(name).EnumerateArray()
        |> Seq.map _.GetDouble()
        |> Seq.toList

    size, cropPct, values "image_mean", values "image_std"

let loadLabels () =
    use document = loadJson "config.json"

    document.RootElement.GetProperty("id2label").EnumerateObject()
    |> Seq.map (fun item -> int item.Name, item.Value.GetString())
    |> Map.ofSeq

let loadImage source =
    match Uri.TryCreate(source, UriKind.Absolute) with
    | true, uri when
        uri.Scheme = Uri.UriSchemeHttp
        || uri.Scheme = Uri.UriSchemeHttps
        ->
        use client = new HttpClient()
        use stream = client.GetStreamAsync(uri).GetAwaiter().GetResult()
        Image.loadStream stream torch.CPU
    | _ -> Image.load source torch.CPU

[<EntryPoint>]
let main argv =
    let source =
        Array.tryItem 0 argv
        |> Option.defaultValue "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/parrots.png"

    printfn "Loading %s at %s ..." repoId revision

    let weightPath =
        Hub.download (hubFile "model.safetensors")
        |> Async.RunSynchronously

    let model = createModel ()
    use reader = SafeTensors.openFile weightPath

    let report =
        ModelState.loadSafeTensorsWith nameMapping Strict (Model.state model) reader

    printfn "Loaded %d parameters and buffers; ignored %d source tensors." report.Loaded.Length report.Ignored.Length

    let size, cropPct, mean, std = loadPreprocessorConfig ()
    let resizeSize = float size / cropPct |> Math.Round |> int
    let labels = loadLabels ()

    scoped {
        let image = loadImage source
        let resize = ResizeShortestEdge.create resizeSize torch.InterpolationMode.Bicubic
        let crop = CenterCrop.create size size
        let normalize: Normalize = { Mean = mean; Std = std }

        let input =
            image
            |> resize.apply
            |> crop.apply
            |> normalize.apply
            |> fun tensor -> tensor.unsqueeze 0L

        let probabilities =
            Toro.noGrad (fun () -> forward model input |> fun logits -> logits.softmax -1L)

        let struct (values, indices) = probabilities.topk 5
        let valueData = values.flatten().data<float32> ()
        let indexData = indices.flatten().data<int64> ()

        printfn "Predictions for %s:" source

        for rank in 0..4 do
            let classIndex = int indexData[rank]
            printfn "  %d. %-30s %.2f%%" (rank + 1) labels[classIndex] (float valueData[rank] * 100.0)
    }

    0
