module CheckpointTests

open System.IO
open System.Text.Json
open Xunit
open FsUnit.Xunit
open Toro
open TorchSharp
open Toro.NN
open TestHelper

let private withTempDir f =
    let dir = Path.Combine(Path.GetTempPath(), $"toro-test-{System.Guid.NewGuid()}")

    try
        f dir
    finally
        if Directory.Exists dir then
            Directory.Delete(dir, true)

let private tensorSum (t: Tensor) = (t.sum ()).ToSingle()

[<Fact>]
let ``Checkpoint round-trip with SGD`` () =
    withTempDir (fun dir ->
        let linear = Linear.init 4 2 torch.float32 torch.CPU
        let vars = Model.trainableParams linear
        let opt = SGD.create 0.05 vars

        let x = torch.randn ([| 8L; 4L |], dtype = torch.float32, device = torch.CPU)
        let target = torch.randn ([| 8L; 2L |], dtype = torch.float32, device = torch.CPU)

        for _ in 1..5 do
            opt.zeroGrad ()

            let loss =
                linear.forward x

                |> fun p -> Loss.mse p target

            loss.backward ()
            opt.step ()

        let weightBefore =
            Model.namedParams linear
            |> List.head
            |> _.Tensor
            |> tensorSum

        Checkpoint.save linear opt 5 dir

        let linear2 = Linear.init 4 2 torch.float32 torch.CPU
        let opt2 = SGD.create 0.01 (Model.trainableParams linear2)

        let epoch = Checkpoint.load linear2 opt2 dir
        epoch |> should equal 5

        let weightAfter =
            Model.namedParams linear2
            |> List.head
            |> _.Tensor
            |> tensorSum

        weightAfter |> should (equalWithin 1e-5f) weightBefore
        opt2.learningRate () |> should (equalWithin 1e-9) 0.05)

[<Fact>]
let ``Checkpoint round-trip with AdamW`` () =
    withTempDir (fun dir ->
        let linear = Linear.init 4 2 torch.float32 torch.CPU
        let vars = Model.trainableParams linear
        let opt = AdamW.createWithLr 0.01 vars

        let x = torch.randn ([| 8L; 4L |], dtype = torch.float32, device = torch.CPU)
        let target = torch.randn ([| 8L; 2L |], dtype = torch.float32, device = torch.CPU)

        for _ in 1..10 do
            opt.zeroGrad ()

            let loss =
                linear.forward x

                |> fun p -> Loss.mse p target

            loss.backward ()
            opt.step ()

        Checkpoint.save linear opt 10 dir

        let linear2 = Linear.init 4 2 torch.float32 torch.CPU

        let opt2 = AdamW.createWithLr 0.001 (Model.trainableParams linear2)


        let epoch = Checkpoint.load linear2 opt2 dir
        epoch |> should equal 10
        opt2.learningRate () |> should (equalWithin 1e-9) 0.01

        let weightOrig =
            Model.namedParams linear
            |> List.head
            |> _.Tensor
            |> tensorSum

        let weightLoaded =
            Model.namedParams linear2
            |> List.head
            |> _.Tensor
            |> tensorSum

        weightLoaded |> should (equalWithin 1e-5f) weightOrig)

[<Fact>]
let ``Checkpoint creates expected directory structure`` () =
    withTempDir (fun dir ->
        let linear = Linear.init 2 1 torch.float32 torch.CPU
        let opt = SGD.create 0.1 (Model.trainableParams linear)

        Checkpoint.save linear opt 1 dir

        File.Exists(Path.Combine(dir, "meta.json"))
        |> should equal true

        File.Exists(Path.Combine(dir, "model.safetensors"))
        |> should equal true

        File.Exists(Path.Combine(dir, "optimizer.safetensors"))
        |> should equal true

        SafeTensors.load (Path.Combine(dir, "optimizer.safetensors"))
        |> Map.isEmpty
        |> should equal true

        use manifest = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "meta.json")))

        manifest.RootElement.GetProperty("FormatVersion").GetInt32()
        |> should equal 2

        manifest.RootElement.GetProperty("OptimizerKind").GetString()
        |> should equal "SGD")

[<Fact>]
let ``AdamW optimizer state survives checkpoint`` () =
    withTempDir (fun dir ->
        let linear = Linear.init 4 2 torch.float32 torch.CPU
        let vars = Model.trainableParams linear
        let opt = AdamW.createWithLr 0.01 vars

        let x = torch.randn ([| 8L; 4L |], dtype = torch.float32, device = torch.CPU)
        let target = torch.randn ([| 8L; 2L |], dtype = torch.float32, device = torch.CPU)

        for _ in 1..5 do
            opt.zeroGrad ()

            let loss =
                linear.forward x

                |> fun p -> Loss.mse p target

            loss.backward ()
            opt.step ()

        Checkpoint.save linear opt 5 dir

        let linear2 = Linear.init 4 2 torch.float32 torch.CPU

        let opt2 = AdamW.createWithLr 0.01 (Model.trainableParams linear2)


        Checkpoint.load linear2 opt2 dir

        |> ignore

        for _ in 6..10 do
            opt.zeroGrad ()

            let loss =
                linear.forward x

                |> fun p -> Loss.mse p target

            loss.backward ()
            opt.step ()

        for _ in 6..10 do
            opt2.zeroGrad ()

            let loss =
                linear2.forward x

                |> fun p -> Loss.mse p target

            loss.backward ()
            opt2.step ()

        let w1 =
            Model.namedParams linear
            |> List.head
            |> _.Tensor
            |> tensorSum

        let w2 =
            Model.namedParams linear2
            |> List.head
            |> _.Tensor
            |> tensorSum

        w2 |> should (equalWithin 1e-4f) w1)

[<Theory>]
[<InlineData("{\"FormatVersion\":1,\"Epoch\":3,\"LearningRate\":0.1,\"OptimizerKind\":\"SGD\"}")>]
[<InlineData("{\"Epoch\":3,\"LearningRate\":0.1,\"OptimizerKind\":\"SGD\"}")>]
let ``Checkpoint rejects old and unversioned manifests before changing the model`` (manifestJson: string) =
    withTempDir (fun dir ->
        let source = Linear.init 4 2 torch.float32 torch.CPU
        let sourceOptimizer = SGD.create 0.1 (Model.trainableParams source)
        Checkpoint.save source sourceOptimizer 3 dir
        File.WriteAllText(Path.Combine(dir, "meta.json"), manifestJson)

        let target = Linear.init 4 2 torch.float32 torch.CPU
        let targetOptimizer = SGD.create 0.2 (Model.trainableParams target)
        let before = tensorSum target.Weight

        Assert.Throws<System.InvalidOperationException>(fun () -> Checkpoint.load target targetOptimizer dir |> ignore)
        |> ignore

        tensorSum target.Weight |> should equal before
        targetOptimizer.learningRate () |> should equal 0.2)

[<Fact>]
let ``Checkpoint rejects optimizer kind mismatch before changing the model`` () =
    withTempDir (fun dir ->
        let source = Linear.init 4 2 torch.float32 torch.CPU
        let sourceOptimizer = AdamW.createWithLr 0.01 (Model.trainableParams source)
        Checkpoint.save source sourceOptimizer 2 dir

        let target = Linear.init 4 2 torch.float32 torch.CPU
        let targetOptimizer = SGD.create 0.2 (Model.trainableParams target)
        let before = tensorSum target.Weight

        Assert.Throws<System.InvalidOperationException>(fun () -> Checkpoint.load target targetOptimizer dir |> ignore)
        |> ignore

        tensorSum target.Weight |> should equal before
        targetOptimizer.learningRate () |> should equal 0.2)

[<Fact>]
let ``Checkpoint validates corrupt optimizer state before changing the model`` () =
    withTempDir (fun dir ->
        let source = Linear.init 4 2 torch.float32 torch.CPU
        let sourceOptimizer = AdamW.createWithLr 0.01 (Model.trainableParams source)
        Checkpoint.save source sourceOptimizer 2 dir

        let optimizerPath = Path.Combine(dir, "optimizer.safetensors")
        let corrupt = SafeTensors.load optimizerPath |> Map.remove "m.Weight"
        SafeTensors.save corrupt optimizerPath

        let target = Linear.init 4 2 torch.float32 torch.CPU
        let targetOptimizer = AdamW.createWithLr 0.2 (Model.trainableParams target)
        let before = tensorSum target.Weight

        Assert.Throws<System.InvalidOperationException>(fun () -> Checkpoint.load target targetOptimizer dir |> ignore)
        |> ignore

        tensorSum target.Weight |> should equal before
        targetOptimizer.learningRate () |> should equal 0.2)
