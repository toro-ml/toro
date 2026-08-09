# SimpleTraining

Train a two-layer MLP on XOR using `sequential { }` and AdamW.

## Run

```bash
dotnet run --project examples/SimpleTraining
```

## Output

```
Training XOR with AdamW...
  epoch  100  loss = ...
  epoch  200  loss = ...
  epoch  300  loss = ...
  epoch  400  loss = ...
  epoch  500  loss = ...

Predictions (expected: 0, 1, 1, 0):
  0 XOR 0 = 0.0xx
  0 XOR 1 = 0.9xx
  1 XOR 0 = 0.9xx
  1 XOR 1 = 0.0xx
```

## Concepts

- `sequential { }` CE for composing `IModule` layers
- `Linear.init`, `Relu` activation
- `AdamW` optimizer, `Loss.mse`
