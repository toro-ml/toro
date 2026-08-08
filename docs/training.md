---
title: Training
category: Documentation
categoryindex: 1
index: 5
---

# Training

This page covers loss functions, optimizers, and the training loop.
For the `result { }` CE and operator return type rules, see [Core Concepts](concepts.html).

## Loss Functions

Toro.NN provides these loss functions. Each takes two tensors and returns `Result<Tensor, ToroError>`:

| Function | Description |
| --- | --- |
| `Loss.mse inp target` | Mean squared error |
| `Loss.nll inp target` | Negative log-likelihood |
| `Loss.crossEntropy inp target` | Cross-entropy (combines log-softmax and NLL) |
| `Loss.binaryCrossEntropyWithLogit inp target` | Binary cross-entropy with logits |

```fsharp
result {
    let! loss = Loss.crossEntropy predictions targets
    printfn "Loss: %.4f" (loss.item ())
}
```

## Optimizers

### SGD

Stochastic Gradient Descent:

```fsharp
let vars = Model.trainableVars model
let sgd = SGD.create 0.01 vars
let opt = sgd :> IOptimizer
```

### AdamW

Adam with weight decay:

```fsharp
result {
    let vars = Model.trainableVars model
    let! adamw = AdamW.createWithLr 0.001 vars
    let opt = adamw :> IOptimizer
}
```

Use `AdamW.create` for full control over hyperparameters:

```fsharp
result {
    let! adamw = AdamW.create {
        ParamsAdamW.defaultParams with
            Lr = 0.001
            Beta1 = 0.9
            Beta2 = 0.999
    } vars
}
```

### IOptimizer Interface

All optimizers implement `IOptimizer`:

```fsharp
type IOptimizer =
    abstract step: unit -> Result<unit, ToroError>
    abstract backwardStep: Tensor -> Result<unit, ToroError>
    abstract learningRate: unit -> float
    abstract setLearningRate: float -> unit
    abstract zeroGrad: unit -> unit
```

- `backwardStep` calls `backward()`, then `step()`, then `zeroGrad()` in one call.
- Use `step` when you need to call `backward()` separately.

## The Training Loop

A typical training loop:

```fsharp
for epoch in 1..epochs do
    for batch in trainLoader do
        let x, y = batch
        let! pred = model.forward x
        let! loss = Loss.crossEntropy pred y
        do! opt.backwardStep loss

    if epoch % 10 = 0 then
        printfn "epoch %d" epoch
```
