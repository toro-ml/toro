module SchedulerTests

open Xunit
open FsUnit.Xunit
open Toro.NN

[<Fact>]
let ``StepDecay decays LR at step boundary`` () =
    let mutable lr = 0.1
    let sched = Scheduler.create (StepDecay(3, 0.5)) (fun v -> lr <- v) 0.1

    lr |> should (equalWithin 1e-9) 0.1

    for _ in 1..3 do
        Scheduler.step sched

    lr |> should (equalWithin 1e-9) 0.05

    for _ in 1..3 do
        Scheduler.step sched

    lr |> should (equalWithin 1e-9) 0.025

[<Fact>]
let ``Exponential decays each step`` () =
    let mutable lr = 1.0
    let sched = Scheduler.create (Exponential 0.9) (fun v -> lr <- v) 1.0

    Scheduler.step sched
    lr |> should (equalWithin 1e-9) 0.9

    Scheduler.step sched
    lr |> should (equalWithin 1e-9) 0.81

[<Fact>]
let ``CosineAnnealing decays at half cycle`` () =
    let mutable lr = 1.0
    let tMax = 100
    let sched = Scheduler.create (CosineAnnealing(tMax, 0.0)) (fun v -> lr <- v) 1.0

    for _ in 1 .. (tMax / 2) do
        Scheduler.step sched

    lr |> should (equalWithin 1e-6) 0.5

[<Fact>]
let ``CosineAnnealing resets after full cycle`` () =
    let mutable lr = 1.0
    let tMax = 10
    let sched = Scheduler.create (CosineAnnealing(tMax, 0.0)) (fun v -> lr <- v) 1.0

    for _ in 1..tMax do
        Scheduler.step sched

    lr |> should (equalWithin 1e-6) 1.0

[<Fact>]
let ``LinearWarmup starts at zero and ramps to base LR`` () =
    let mutable lr = 0.01
    let sched = Scheduler.create (LinearWarmup 5) (fun v -> lr <- v) 0.01

    lr |> should (equalWithin 1e-9) 0.0
    Scheduler.currentLr sched |> should (equalWithin 1e-9) 0.0

    Scheduler.step sched
    lr |> should (equalWithin 1e-9) 0.002

    for _ in 2..5 do
        Scheduler.step sched

    lr |> should (equalWithin 1e-9) 0.01

[<Fact>]
let ``LinearWarmup holds LR after warmup`` () =
    let mutable lr = 0.01
    let sched = Scheduler.create (LinearWarmup 2) (fun v -> lr <- v) 0.01

    for _ in 1..5 do
        Scheduler.step sched

    lr |> should (equalWithin 1e-9) 0.01

[<Fact>]
let ``LrSchedule.lrAt is pure`` () =
    let lr1 = LrSchedule.lrAt 1.0 (Exponential 0.5) 3
    let lr2 = LrSchedule.lrAt 1.0 (Exponential 0.5) 3
    lr1 |> should (equalWithin 1e-15) lr2
    lr1 |> should (equalWithin 1e-9) 0.125

[<Fact>]
let ``LinearWarmupDecay ramps then decays to end LR`` () =
    let mutable lr = 1.0
    let sched = Scheduler.create (LinearWarmupDecay(2, 6, 0.0)) (fun v -> lr <- v) 1.0

    lr |> should (equalWithin 1e-9) 0.0

    Scheduler.step sched
    lr |> should (equalWithin 1e-9) 0.5

    Scheduler.step sched
    lr |> should (equalWithin 1e-9) 1.0

    Scheduler.step sched
    lr |> should (equalWithin 1e-9) 0.75

    Scheduler.step sched
    lr |> should (equalWithin 1e-9) 0.5

    Scheduler.step sched
    lr |> should (equalWithin 1e-9) 0.25

    Scheduler.step sched
    lr |> should (equalWithin 1e-9) 0.0
