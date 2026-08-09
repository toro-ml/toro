namespace Toro.NN

/// Learning-rate scheduler interface.
type IScheduler =
    /// Advance one step and update the optimizer learning rate.
    abstract step: unit -> unit
    /// Return the current learning rate.
    abstract currentLr: unit -> float

/// Step-decay scheduler: multiply LR by gamma every stepSize steps.
type StepLR = {
    Optimizer: IOptimizer
    StepSize: int
    Gamma: float
    BaseLr: float
    mutable CurrentStep: int
} with

    interface IScheduler with
        member this.step() =
            this.CurrentStep <- this.CurrentStep + 1

            let lr =
                this.BaseLr
                * pown this.Gamma (this.CurrentStep / this.StepSize)

            this.Optimizer.setLearningRate lr

        member this.currentLr() = this.Optimizer.learningRate ()

module StepLR =
    /// Create a step-decay scheduler.
    let create (stepSize: int) (gamma: float) (opt: IOptimizer) : IScheduler = {
        Optimizer = opt
        StepSize = stepSize
        Gamma = gamma
        BaseLr = opt.learningRate ()
        CurrentStep = 0
    }

/// Exponential-decay scheduler: multiply LR by gamma each step.
type ExponentialLR = {
    Optimizer: IOptimizer
    Gamma: float
    BaseLr: float
    mutable CurrentStep: int
} with

    interface IScheduler with
        member this.step() =
            this.CurrentStep <- this.CurrentStep + 1
            let lr = this.BaseLr * pown this.Gamma this.CurrentStep
            this.Optimizer.setLearningRate lr

        member this.currentLr() = this.Optimizer.learningRate ()

module ExponentialLR =
    /// Create an exponential-decay scheduler.
    let create (gamma: float) (opt: IOptimizer) : IScheduler = {
        Optimizer = opt
        Gamma = gamma
        BaseLr = opt.learningRate ()
        CurrentStep = 0
    }

/// Cosine-annealing scheduler: decay LR following a cosine curve to etaMin over tMax steps, then reset.
type CosineAnnealingLR = {
    Optimizer: IOptimizer
    TMax: int
    EtaMin: float
    BaseLr: float
    mutable CurrentStep: int
} with

    interface IScheduler with
        member this.step() =
            this.CurrentStep <- this.CurrentStep + 1
            let t = float (this.CurrentStep % this.TMax) / float this.TMax

            let lr =
                this.EtaMin
                + 0.5
                  * (this.BaseLr - this.EtaMin)
                  * (1.0 + cos (System.Math.PI * t))

            this.Optimizer.setLearningRate lr

        member this.currentLr() = this.Optimizer.learningRate ()

module CosineAnnealingLR =
    /// Create a cosine-annealing scheduler.
    let create (tMax: int) (etaMin: float) (opt: IOptimizer) : IScheduler = {
        Optimizer = opt
        TMax = tMax
        EtaMin = etaMin
        BaseLr = opt.learningRate ()
        CurrentStep = 0
    }

/// Linear warmup followed by a constant LR.
type LinearWarmup = {
    Optimizer: IOptimizer
    WarmupSteps: int
    BaseLr: float
    mutable CurrentStep: int
} with

    interface IScheduler with
        member this.step() =
            this.CurrentStep <- this.CurrentStep + 1

            if this.CurrentStep <= this.WarmupSteps then
                let lr =
                    this.BaseLr * float this.CurrentStep
                    / float this.WarmupSteps

                this.Optimizer.setLearningRate lr

        member this.currentLr() = this.Optimizer.learningRate ()

module LinearWarmup =
    /// Create a linear-warmup scheduler.
    let create (warmupSteps: int) (opt: IOptimizer) : IScheduler = {
        Optimizer = opt
        WarmupSteps = warmupSteps
        BaseLr = opt.learningRate ()
        CurrentStep = 0
    }
