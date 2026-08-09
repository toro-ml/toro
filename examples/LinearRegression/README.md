# LinearRegression

Learn `y = 3x + 2` by manual gradient descent using raw tensors — no `Toro.NN` layers or optimizers.

## Run

```bash
dotnet run --project examples/LinearRegression
```

## Output

```
Linear regression: learning y = 3x + 2
lr = 0.1, steps = 200

  step   1  loss=...  w=...  b=...
  step  50  loss=...  w=...  b=...
  step 100  loss=...  w=...  b=...
  step 150  loss=...  w=...  b=...
  step 200  loss=...  w=...  b=...

Learned:  y = 3.0xxx * x + 2.0xxx
Expected: y = 3.0000 * x + 2.0000
```

## Concepts

- `Tensor.randn`, `requiresGrad`, `backward`, `grad`, `copyInPlace`
- Manual SGD update loop without an optimizer
