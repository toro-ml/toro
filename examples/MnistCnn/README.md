# MnistCnn

Train a CNN on MNIST with BatchNorm, Dropout, and `sequentialT { }`.

## Run

```bash
dotnet run --project examples/MnistCnn
```

MNIST is downloaded automatically on first run.

## Concepts

- `sequentialT { }` CE for train/eval-aware layers (`IModuleT`)
- `BatchNorm`, `Dropout`, `MaxPool2d`
- `forwardT x true` for training, `forwardT x false` for evaluation
