# MnistCnn

Train a CNN on MNIST with BatchNorm, Dropout, and Kleisli composition (`>=>`).

## Run

```bash
dotnet run --project examples/MnistCnn
```

MNIST is downloaded automatically on first run.

## Concepts

- Kleisli composition (`>=>`) for composing layers and train-aware functions
- `BatchNorm`, `Dropout`, `MaxPool2d`
- Partial application: `drop.forwardT train` yields `Tensor -> Result<Tensor, ToroError>`
