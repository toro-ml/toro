namespace Toro.NN

/// Discriminated union describing learning-rate schedule shapes.
type LrSchedule =
    | StepDecay of stepSize: int * gamma: float
    | Exponential of gamma: float
    | CosineAnnealing of tMax: int * etaMin: float
    | LinearWarmup of warmupSteps: int

module LrSchedule =
    /// Pure function: compute the learning rate at a given step.
    let lrAt (baseLr: float) (schedule: LrSchedule) (step: int) : float =
        match schedule with
        | StepDecay(stepSize, gamma) -> baseLr * pown gamma (step / stepSize)
        | Exponential gamma -> baseLr * pown gamma step
        | CosineAnnealing(tMax, etaMin) ->
            let t = float (step % tMax) / float tMax

            etaMin
            + 0.5 * (baseLr - etaMin) * (1.0 + cos (System.Math.PI * t))
        | LinearWarmup warmupSteps ->
            if step = 0 then baseLr
            elif step <= warmupSteps then baseLr * float step / float warmupSteps
            else baseLr

/// Scheduler that pairs a schedule with mutable step counter and LR setter.
type Scheduler = {
    Schedule: LrSchedule
    BaseLr: float
    mutable CurrentStep: int
    SetLr: float -> unit
}

module Scheduler =
    let create (schedule: LrSchedule) (setLr: float -> unit) (baseLr: float) : Scheduler = {
        Schedule = schedule
        BaseLr = baseLr
        CurrentStep = 0
        SetLr = setLr
    }

    let step (sched: Scheduler) =
        sched.CurrentStep <- sched.CurrentStep + 1
        let lr = LrSchedule.lrAt sched.BaseLr sched.Schedule sched.CurrentStep
        sched.SetLr lr

    let currentLr (sched: Scheduler) =
        LrSchedule.lrAt sched.BaseLr sched.Schedule sched.CurrentStep
