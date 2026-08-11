module CheckpointTests

open System.IO
open Xunit
open FsUnit.Xunit
open Toro
open Toro.NN
open TestHelper

let private withTempDir f =
    let dir = Path.Combine(Path.GetTempPath(), $"toro-test-{System.Guid.NewGuid()}")

    try
        f dir
    finally
        if Directory.Exists dir then
            Directory.Delete(dir, true)

let private tensorSum (t: Tensor) = (t.sumAll ()).toFloat32Scalar ()

[<Fact>]
let ``Checkpoint round-trip with SGD`` () =
    withTempDir (fun dir ->
        let linear = Linear.init 4 2 F32 Cpu
        let vars = Model.trainableVars linear
        let opt = SGD.create 0.05 vars

        let x = Tensor.randn ([ 8; 4 ], F32, Cpu)
        let target = Tensor.randn ([ 8; 2 ], F32, Cpu)

        for _ in 1..5 do
            opt.zeroGrad ()

            let loss =
                linear.forward x

                |> fun p -> Loss.mse p target

            loss.backward ()
            opt.step ()

        let weightBefore = Model.namedParams linear |> List.head |> snd |> tensorSum

        Checkpoint.save linear (opt.toOps ()) 5 dir

        let linear2 = Linear.init 4 2 F32 Cpu
        let opt2 = SGD.create 0.01 (Model.trainableVars linear2)

        let epoch = Checkpoint.load linear2 (opt2.toOps ()) dir
        epoch |> should equal 5

        let weightAfter = Model.namedParams linear2 |> List.head |> snd |> tensorSum
        weightAfter |> should (equalWithin 1e-5f) weightBefore
        opt2.learningRate () |> should (equalWithin 1e-9) 0.05)

[<Fact>]
let ``Checkpoint round-trip with AdamW`` () =
    withTempDir (fun dir ->
        let linear = Linear.init 4 2 F32 Cpu
        let vars = Model.trainableVars linear
        let opt = AdamW.createWithLr 0.01 vars

        let x = Tensor.randn ([ 8; 4 ], F32, Cpu)
        let target = Tensor.randn ([ 8; 2 ], F32, Cpu)

        for _ in 1..10 do
            opt.zeroGrad ()

            let loss =
                linear.forward x

                |> fun p -> Loss.mse p target

            loss.backward ()
            opt.step ()

        Checkpoint.save linear (opt.toOps ()) 10 dir

        let linear2 = Linear.init 4 2 F32 Cpu

        let opt2 = AdamW.createWithLr 0.001 (Model.trainableVars linear2)


        let epoch = Checkpoint.load linear2 (opt2.toOps ()) dir
        epoch |> should equal 10
        opt2.learningRate () |> should (equalWithin 1e-9) 0.01

        let weightOrig = Model.namedParams linear |> List.head |> snd |> tensorSum
        let weightLoaded = Model.namedParams linear2 |> List.head |> snd |> tensorSum
        weightLoaded |> should (equalWithin 1e-5f) weightOrig)

[<Fact>]
let ``Checkpoint creates expected directory structure`` () =
    withTempDir (fun dir ->
        let linear = Linear.init 2 1 F32 Cpu
        let opt = SGD.create 0.1 (Model.trainableVars linear)

        Checkpoint.save linear (opt.toOps ()) 1 dir

        File.Exists(Path.Combine(dir, "meta.json"))
        |> should equal true

        File.Exists(Path.Combine(dir, "model.safetensors"))
        |> should equal true)

[<Fact>]
let ``AdamW optimizer state survives checkpoint`` () =
    withTempDir (fun dir ->
        let linear = Linear.init 4 2 F32 Cpu
        let vars = Model.trainableVars linear
        let opt = AdamW.createWithLr 0.01 vars

        let x = Tensor.randn ([ 8; 4 ], F32, Cpu)
        let target = Tensor.randn ([ 8; 2 ], F32, Cpu)

        for _ in 1..5 do
            opt.zeroGrad ()

            let loss =
                linear.forward x

                |> fun p -> Loss.mse p target

            loss.backward ()
            opt.step ()

        Checkpoint.save linear (opt.toOps ()) 5 dir

        let linear2 = Linear.init 4 2 F32 Cpu

        let opt2 = AdamW.createWithLr 0.01 (Model.trainableVars linear2)


        Checkpoint.load linear2 (opt2.toOps ()) dir

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

        let w1 = Model.namedParams linear |> List.head |> snd |> tensorSum
        let w2 = Model.namedParams linear2 |> List.head |> snd |> tensorSum
        w2 |> should (equalWithin 1e-4f) w1)
