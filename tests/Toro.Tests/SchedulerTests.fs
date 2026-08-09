module SchedulerTests

open Xunit
open FsUnit.Xunit
open Toro
open Toro.NN
open TestHelper

[<Fact>]
let ``StepLR decays LR at step boundary`` () =
    let opt = SGD.create 0.1 []
    let sched = StepLR.create 3 0.5 opt

    opt.learningRate () |> should (equalWithin 1e-9) 0.1

    for _ in 1..3 do
        sched.step ()

    opt.learningRate () |> should (equalWithin 1e-9) 0.05

    for _ in 1..3 do
        sched.step ()

    opt.learningRate () |> should (equalWithin 1e-9) 0.025

[<Fact>]
let ``ExponentialLR decays each step`` () =
    let opt = SGD.create 1.0 []
    let sched = ExponentialLR.create 0.9 opt

    sched.step ()
    opt.learningRate () |> should (equalWithin 1e-9) 0.9

    sched.step ()
    opt.learningRate () |> should (equalWithin 1e-9) 0.81

[<Fact>]
let ``CosineAnnealingLR decays at half cycle`` () =
    let opt = SGD.create 1.0 []
    let tMax = 100
    let sched = CosineAnnealingLR.create tMax 0.0 opt

    for _ in 1 .. (tMax / 2) do
        sched.step ()

    opt.learningRate () |> should (equalWithin 1e-6) 0.5

[<Fact>]
let ``CosineAnnealingLR resets after full cycle`` () =
    let opt = SGD.create 1.0 []
    let tMax = 10
    let sched = CosineAnnealingLR.create tMax 0.0 opt

    for _ in 1..tMax do
        sched.step ()

    opt.learningRate () |> should (equalWithin 1e-6) 1.0

[<Fact>]
let ``LinearWarmup ramps to base LR`` () =
    let opt = SGD.create 0.01 []
    let sched = LinearWarmup.create 5 opt

    sched.step ()
    opt.learningRate () |> should (equalWithin 1e-9) 0.002

    for _ in 2..5 do
        sched.step ()

    opt.learningRate () |> should (equalWithin 1e-9) 0.01

[<Fact>]
let ``LinearWarmup holds LR after warmup`` () =
    let opt = SGD.create 0.01 []
    let sched = LinearWarmup.create 2 opt

    for _ in 1..5 do
        sched.step ()

    opt.learningRate () |> should (equalWithin 1e-9) 0.01
