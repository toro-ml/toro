namespace Toro.NN

/// Discriminated union describing learning-rate schedule shapes.
type LrSchedule =
    | StepDecay of stepSize: int * gamma: float
    | Exponential of gamma: float
    | CosineAnnealing of tMax: int * etaMin: float
    | LinearWarmup of warmupSteps: int
    | OneCycle of totalSteps: int * maxLr: float * divFactor: float * finalDivFactor: float * pctStart: float
    | CosineAnnealingWarmRestarts of t0: int * tMult: int * etaMin: float
    | Polynomial of totalSteps: int * power: float * endLr: float
    | CyclicLR of baseLr: float * maxLr: float * stepSizeUp: int * stepSizeDown: int

module LrSchedule =

    let rec private restartPosition multiplier position cycleLength =
        if position < cycleLength then
            position, cycleLength
        else
            restartPosition multiplier (position - cycleLength) (cycleLength * multiplier)

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
            if step = 0 then
                baseLr
            elif step <= warmupSteps then
                baseLr * float step / float warmupSteps
            else
                baseLr
        | OneCycle(totalSteps, maxLr, divFactor, finalDivFactor, pctStart) ->
            let warmup = int (float totalSteps * pctStart)
            let initLr = maxLr / divFactor
            let minLr = initLr / finalDivFactor

            if step <= warmup then
                let pct = float step / float warmup
                initLr + (maxLr - initLr) * pct
            else
                let pct = float (step - warmup) / float (totalSteps - warmup)
                let cosVal = cos (System.Math.PI * pct)
                minLr + 0.5 * (maxLr - minLr) * (1.0 + cosVal)
        | CosineAnnealingWarmRestarts(t0, tMult, etaMin) ->
            let position, cycleLength = restartPosition tMult step t0
            let t = float position / float cycleLength

            etaMin
            + 0.5 * (baseLr - etaMin) * (1.0 + cos (System.Math.PI * t))
        | Polynomial(totalSteps, power, endLr) ->
            if step >= totalSteps then
                endLr
            else
                let decay = (1.0 - float step / float totalSteps) ** power
                (baseLr - endLr) * decay + endLr
        | CyclicLR(cyclicBase, maxLr, stepSizeUp, stepSizeDown) ->
            let cycleLen = stepSizeUp + stepSizeDown
            let pos = step % cycleLen

            if pos < stepSizeUp then
                let pct = float pos / float stepSizeUp
                cyclicBase + (maxLr - cyclicBase) * pct
            else
                let pct = float (pos - stepSizeUp) / float stepSizeDown
                maxLr - (maxLr - cyclicBase) * pct

/// Scheduler that pairs a schedule with mutable step counter and LR setter.
type Scheduler = {
    Schedule: LrSchedule
    BaseLr: float
    mutable CurrentStep: int
    SetLr: float -> unit
}

/// Mutable state snapshot of a learning-rate scheduler.
type SchedulerState = { CurrentStep: int }

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

    /// Capture the mutable scheduler state without its schedule configuration or LR setter.
    let getState (sched: Scheduler) : SchedulerState = { CurrentStep = sched.CurrentStep }

    /// Restore mutable scheduler state and apply the corresponding learning rate.
    let loadState (state: SchedulerState) (sched: Scheduler) : unit =
        if state.CurrentStep < 0 then
            invalidArg (nameof state) $"Scheduler step must be non-negative, but is {state.CurrentStep}."

        sched.CurrentStep <- state.CurrentStep
        sched.SetLr(currentLr sched)
