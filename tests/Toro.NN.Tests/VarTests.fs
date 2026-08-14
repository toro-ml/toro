module VarTests

open System
open System.Collections.Generic
open System.IO
open Xunit
open FsUnit.Xunit
open Toro
open TorchSharp
open Toro.NN
open TestHelper

type ParamBranch = TwoLayers of first: Linear * second: Linear

type ClassifiedState = {
    [<Parameter>]
    Trainable: Tensor
    [<Parameter>]
    Frozen: Tensor
    [<Buffer>]
    Running: Tensor
    [<ModelIgnore>]
    Ignored: Tensor
}

type UnclassifiedState = { Value: Tensor }

type ConflictingAttributes = {
    [<Parameter; Buffer>]
    Value: Tensor
}

type SharedParameters = {
    [<Parameter>]
    First: Tensor
    [<Parameter>]
    Second: Tensor
}

type SharedRoles = {
    [<Parameter>]
    Parameter: Tensor
    [<Buffer>]
    Buffer: Tensor
}

type TensorContainers = {
    [<Parameter>]
    Optional: Tensor option
    [<Parameter>]
    Pair: Tensor * Tensor
}

let private parameter value requiresGrad =
    let tensor = Init.toTensor [| 2L |] torch.float32 torch.CPU (Init.Const value)
    tensor.requires_grad <- requiresGrad
    tensor

let private names (tensors: NamedTensor list) = tensors |> List.map _.Name
let private tensorSum (tensor: Tensor) = (tensor.sum ()).ToSingle()

let private withTempDir action =
    let dir = Path.Combine(Path.GetTempPath(), $"toro-model-{Guid.NewGuid()}")

    try
        Directory.CreateDirectory dir |> ignore
        action dir
    finally
        if Directory.Exists dir then
            Directory.Delete(dir, true)

[<Fact>]
let ``named state classifies parameters buffers frozen values and ignored values`` () =
    let model = {
        Trainable = parameter 1.0 true
        Frozen = parameter 2.0 false
        Running = parameter 3.0 false
        Ignored = parameter 4.0 true
    }

    Model.namedState model
    |> List.map (fun item -> item.Name, item.Kind)
    |> should equal [ "Trainable", Parameter; "Frozen", Parameter; "Running", Buffer ]

    Model.namedParams model
    |> names
    |> should equal [ "Trainable"; "Frozen" ]

    Model.namedBuffers model
    |> names
    |> should equal [ "Running" ]

    Model.trainableParams model
    |> names
    |> should equal [ "Trainable" ]

[<Fact>]
let ``named state rejects unclassified tensors with their path`` () =
    let model = { Value = parameter 1.0 true }

    let error =
        Assert.Throws<InvalidOperationException>(fun () -> Model.namedState model |> ignore)

    Assert.Contains("Value", error.Message)

[<Fact>]
let ``named state rejects conflicting field attributes`` () =
    let model: ConflictingAttributes = { Value = parameter 1.0 true }

    let error =
        Assert.Throws<InvalidOperationException>(fun () -> Model.namedState model |> ignore)

    Assert.Contains("Value", error.Message)

[<Fact>]
let ``named state canonicalizes shared tensors to the first path`` () =
    let shared = parameter 1.0 true
    let model = { First = shared; Second = shared }
    Model.namedState model |> names |> should equal [ "First" ]

[<Fact>]
let ``named state rejects conflicting roles for a shared tensor`` () =
    let shared = parameter 1.0 true

    let model: SharedRoles = { Parameter = shared; Buffer = shared }

    let error =
        Assert.Throws<InvalidOperationException>(fun () -> Model.namedState model |> ignore)

    Assert.Contains("Parameter", error.Message)
    Assert.Contains("Buffer", error.Message)

[<Fact>]
let ``namedParams keeps established record option tuple union and list names`` () =
    let model = {|
        Branch = TwoLayers(Linear.initNoBias 3 2 torch.float32 torch.CPU, Linear.initNoBias 2 1 torch.float32 torch.CPU)
        Optional = Some(Linear.initNoBias 4 3 torch.float32 torch.CPU)
        Pair = Linear.initNoBias 2 1 torch.float32 torch.CPU, Linear.initNoBias 1 1 torch.float32 torch.CPU
        Layers = [ Linear.initNoBias 1 1 torch.float32 torch.CPU ]
    |}

    Model.namedParams model
    |> names
    |> should equal [
        "Branch.TwoLayers.0.Weight"
        "Branch.TwoLayers.1.Weight"
        "Layers.0.Weight"
        "Optional.Weight"
        "Pair.0.Weight"
        "Pair.1.Weight"
    ]

[<Fact>]
let ``parameter attributes propagate through option and tuple containers`` () =
    let model: TensorContainers = {
        Optional = Some(parameter 1.0 true)
        Pair = parameter 2.0 true, parameter 3.0 true
    }

    Model.namedParams model
    |> names
    |> should equal [ "Optional"; "Pair.0"; "Pair.1" ]

[<Fact>]
let ``namedParams preserves Sequential layer names and order`` () =
    let model =
        sequential {
            Linear.initNoBias 4 2 torch.float32 torch.CPU
            Linear.initNoBias 2 1 torch.float32 torch.CPU
        }

    Model.namedParams model
    |> names
    |> should equal [ "Layers.0.Weight"; "Layers.1.Weight" ]

[<Fact>]
let ``namedParams sorts string dictionary keys ordinally`` () =
    let dictionary = Dictionary<string, Linear>()
    dictionary.Add("z", Linear.initNoBias 2 1 torch.float32 torch.CPU)
    dictionary.Add("A", Linear.initNoBias 4 2 torch.float32 torch.CPU)

    Model.namedParams dictionary
    |> names
    |> should equal [ "A.Weight"; "z.Weight" ]

[<Fact>]
let ``named state rejects unstable enumerables and invalid dictionaries`` () =
    let sequence = seq { Linear.initNoBias 2 1 torch.float32 torch.CPU }

    Assert.Throws<InvalidOperationException>(fun () -> Model.namedState sequence |> ignore)
    |> ignore

    let nonString = Dictionary<int, Linear>()
    nonString.Add(0, Linear.initNoBias 2 1 torch.float32 torch.CPU)

    Assert.Throws<InvalidOperationException>(fun () -> Model.namedState nonString |> ignore)
    |> ignore

    let dotted = Map [ "invalid.name", Linear.initNoBias 2 1 torch.float32 torch.CPU ]

    Assert.Throws<InvalidOperationException>(fun () -> Model.namedState dotted |> ignore)
    |> ignore

[<Fact>]
let ``named state rejects cycles with the active path`` () =
    let items = ResizeArray<obj>()
    items.Add items

    let error =
        Assert.Throws<InvalidOperationException>(fun () -> Model.namedState items |> ignore)

    Assert.Contains("0", error.Message)

[<Fact>]
let ``type plan caches do not change discovery results`` () =
    let first = Linear.init 4 2 torch.float32 torch.CPU
    let second = Linear.init 4 2 torch.float32 torch.CPU

    Model.namedState first
    |> names
    |> should equal (Model.namedState second |> names)

[<Fact>]
let ``BatchNorm exposes affine values as parameters and running state as buffers`` () =
    let batchNorm = BatchNorm.initDefault 4 torch.float32 torch.CPU

    Model.namedParams batchNorm
    |> names
    |> should equal [ "Weight"; "Bias" ]

    Model.namedBuffers batchNorm
    |> names
    |> should equal [ "RunningMean"; "RunningVar" ]

[<Fact>]
let ``Model save and load round-trips parameters and buffers`` () =
    withTempDir (fun dir ->
        let path = Path.Combine(dir, "model.safetensors")
        let source = BatchNorm.initDefault 4 torch.float32 torch.CPU
        source.RunningMean.copyInPlace (torch.full_like (source.RunningMean, 7.0))
        Model.save source path

        SafeTensors.loadMeta path
        |> Map.keys
        |> Seq.toList
        |> should equal [ "Bias"; "RunningMean"; "RunningVar"; "Weight" ]

        let target = BatchNorm.initDefault 4 torch.float32 torch.CPU
        let report = Model.loadInto target path Strict

        report.Loaded
        |> should equal [ "Weight"; "Bias"; "RunningMean"; "RunningVar" ]

        tensorSum target.RunningMean |> should equal 28.0f)

[<Fact>]
let ``Strict load validates all tensors before changing any tensor`` () =
    let model = Linear.init 3 2 torch.float32 torch.CPU
    let beforeWeight = tensorSum model.Weight
    let replacementWeight = torch.full_like (model.Weight, 9.0)
    let invalidBias = torch.zeros ([| 3L |], dtype = torch.float32)

    Assert.Throws<InvalidOperationException>(fun () ->
        Map [ "Weight", replacementWeight; "Bias", invalidBias ]
        |> Model.loadFromDict Strict model
        |> ignore)
    |> ignore

    tensorSum model.Weight |> should equal beforeWeight

[<Fact>]
let ``Lenient load changes only matching canonical state`` () =
    let model = Linear.init 3 2 torch.float32 torch.CPU
    let beforeBias = tensorSum model.Bias.Value
    let replacementWeight = torch.full_like (model.Weight, 2.0)
    let invalidBias = torch.zeros ([| 3L |], dtype = torch.float32)

    let report =
        Map [ "Weight", replacementWeight; "Bias", invalidBias ]
        |> Model.loadFromDict Lenient model

    report.Loaded |> should equal [ "Weight" ]

    report.ShapeMismatches
    |> List.map _.Name
    |> should equal [ "Bias" ]

    tensorSum model.Weight |> should equal 12.0f
    tensorSum model.Bias.Value |> should equal beforeBias

[<Fact>]
let ``NameMapping collisions are rejected before model changes`` () =
    let model = Linear.initNoBias 2 2 torch.float32 torch.CPU
    let before = tensorSum model.Weight
    let source = torch.ones_like model.Weight
    let tensors = Map [ "first", source; "second", source ]

    let mapping =
        NameMapping.create [ NameRule.rename "first" "Weight"; NameRule.rename "second" "Weight" ]

    let error =
        Assert.Throws<InvalidOperationException>(fun () ->
            tensors
            |> Model.loadFromDictWith mapping Strict model
            |> ignore)

    Assert.Contains("first", error.Message)
    Assert.Contains("second", error.Message)
    tensorSum model.Weight |> should equal before

[<Fact>]
let ``NameMapping can intentionally ignore source suffixes in Strict mode`` () =
    let model = Linear.initNoBias 2 2 torch.float32 torch.CPU
    let replacement = torch.ones_like model.Weight
    let externalState = torch.zeros ([| 1L |], dtype = torch.int64)

    let mapping = NameMapping.create [ NameRule.ignoreSuffix "num_batches_tracked" ]

    let report =
        Map [ "Weight", replacement; "external.norm.num_batches_tracked", externalState ]
        |> Model.loadFromDictWith mapping Strict model

    report.Loaded |> should equal [ "Weight" ]

    report.Ignored
    |> should equal [ "external.norm.num_batches_tracked" ]

    report.Unexpected |> should be Empty

[<Fact>]
let ``Ambiguous NameMapping rules fail before model changes`` () =
    let model = Linear.initNoBias 2 2 torch.float32 torch.CPU
    let before = tensorSum model.Weight
    let replacement = torch.ones_like model.Weight

    let mapping =
        NameMapping.create [ NameRule.rename "external" "Weight"; NameRule.rewrite "{name}" "{name}" ]

    let error =
        Assert.Throws<InvalidOperationException>(fun () ->
            Map [ "external", replacement ]
            |> Model.loadFromDictWith mapping Strict model
            |> ignore)

    Assert.Contains("matches multiple", error.Message)
    tensorSum model.Weight |> should equal before

[<Fact>]
let ``NameRule rejects references to undeclared captures`` () =
    let error =
        Assert.Throws<ArgumentException>(fun () ->
            NameRule.rewrite "layer.{index}.weight" "Layers.{missing}.Weight"
            |> ignore)

    Assert.Contains("unknown capture", error.Message)

[<Fact>]
let ``Model save writes a shared tensor only under its canonical name`` () =
    withTempDir (fun dir ->
        let path = Path.Combine(dir, "model.safetensors")
        let shared = parameter 4.0 true
        let model = { First = shared; Second = shared }
        Model.save model path

        SafeTensors.loadMeta path
        |> Map.keys
        |> Seq.toList
        |> should equal [ "First" ])

// --- Init tests ---

[<Fact>]
let ``Uniform init produces values in range`` () =
    let lo, up = -1.0, 1.0

    let tensor =
        Init.toTensor [| 10000L |] torch.float64 torch.CPU (Init.Uniform(lo, up))

    let mean = (tensor.mean ()).ToDouble()
    mean |> should be (greaterThan (lo + 0.1))
    mean |> should be (lessThan (up - 0.1))

[<Fact>]
let ``KaimingNormal init has reasonable variance`` () =
    let shape = [| 256L; 128L |]
    let expectedStd = sqrt (2.0 / 128.0)
    let tensor = Init.toTensor shape torch.float32 torch.CPU Init.KaimingNormal
    let mean = (tensor.mean ()).ToSingle()
    let meanSquare = (tensor.square().mean ()).ToSingle()
    let actualStd = sqrt (float meanSquare - float mean * float mean)
    abs (actualStd - expectedStd) |> should be (lessThan 0.02)

[<Fact>]
let ``Init Const creates tensor with given value`` () =
    let tensor = Init.toTensor [| 3L; 2L |] torch.float32 torch.CPU (Init.Const 5.0)
    tensor.shape |> should equal [| 3L; 2L |]
    tensorSum tensor |> should equal 30.0f

[<Fact>]
let ``Init Randn creates tensor with specified mean`` () =
    let tensor =
        Init.toTensor [| 10000L |] torch.float64 torch.CPU (Init.Randn(3.0, 0.01))

    (tensor.mean ()).ToDouble() |> should (equalWithin 0.1) 3.0
