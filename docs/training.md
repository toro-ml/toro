---
title: Training
category: Documentation
categoryindex: 1
index: 4
---

# Training

This page covers loss functions, optimizers, the training loop, and the `result { }` pattern.

## Loss Functions

Toro.NN provides these loss functions:

| Function | Description |
| --- | --- |
| `Loss.mse inp target` | Mean squared error |
| `Loss.nll inp target` | Negative log-likelihood |
| `Loss.crossEntropy inp target` | Cross-entropy (combines log-softmax and NLL) |
| `Loss.binaryCrossEntropyWithLogit inp target` | Binary cross-entropy with logits |

Each function takes two tensors and returns `Result<Tensor, ToroError>`.

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

## The `result { }` Pattern

Toro uses `Result<'T, ToroError>` to report errors from tensor and module operations.
The `result { }` computation expression chains these operations.

### Operator Categories

**Operators that return `Tensor` directly** (throw on error):

- Arithmetic: `+`, `-`, `*`, `/`, unary `-`
- Comparison: `.=.`, `.<>.`, `.>.`, `.<.`, `.>=.`, `.<=.`

**Methods that return `Result<Tensor, ToroError>`**:

- Factory methods: `Tensor.zeros`, `Tensor.randn`, etc.
- Shape operations: `reshape`, `view`, `squeeze`, `unsqueeze`, etc.
- Math operations: `matmul`, `softmax`, `exp`, `log`, etc.
- Reduction operations: `sumAll`, `meanAll`, `sum`, `mean`, etc.
- Module forward: `model.forward`, `model.forwardT`

### Usage in `result { }`

Use `let!` for `Result`-returning operations and `let` for direct values:

```fsharp
let r = result {
    let! x = Tensor.randn ([ 4; 2 ], F32, Cpu)    // Result -> use let!
    let! w = Tensor.randn ([ 2; 1 ], F32, Cpu)     // Result -> use let!
    let! pred = x.matmul w                          // Result -> use let!
    let shifted = pred + 1.0                        // Tensor -> use let
    let! loss = shifted.meanAll ()                   // Result -> use let!
    do! loss.backward ()                             // Result<unit> -> use do!
}

match r with
| Ok () -> printfn "Success"
| Error e -> eprintfn "Error: %A" e
```

## Disabling Gradients

Use `Toro.noGrad` to disable gradient tracking during evaluation:

```fsharp
Toro.noGrad (fun () ->
    let r = result {
        let! pred = model.forward testX
        let! loss = Loss.crossEntropy pred testY
        printfn "Test loss: %.4f" (loss.item ())
    }

    match r with
    | Ok () -> ()
    | Error e -> eprintfn "%A" e
)
```

`Toro.noGrad` wraps a function call in a `torch.no_grad()` scope.
This reduces memory use and speeds up inference.
