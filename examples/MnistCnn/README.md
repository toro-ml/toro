# MnistCnn

Train a CNN on MNIST with BatchNorm and Dropout.

## Run

```bash
dotnet run --project examples/MnistCnn
```

MNIST is downloaded automatically on first run.

## Concepts

- `pipeline { }` CE for composing layers and train-aware functions
- `BatchNorm`, `Dropout`, `MaxPool2d`
- Partial application: `drop.forwardT train` yields `Tensor -> Tensor`
