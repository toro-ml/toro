# MnistTraining

Train a CNN on MNIST using an F# record model with `Conv2d` and `Linear` layers.

## Run

```bash
dotnet run --project examples/MnistTraining
```

MNIST is downloaded automatically on first run.

## Concepts

- F# record implementing `IModule`
- `Conv2d.init` with custom stride config
- Training loop with `DataLoader` batches
- Train/test accuracy evaluation with `Toro.noGrad`
