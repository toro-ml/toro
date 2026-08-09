# SimpleGcn

GCN node classification on a synthetic 8-node graph with two clusters.

## Run

```bash
dotnet run --project examples/SimpleGcn
```

## Concepts

- `GCNConv` layers from `Toro.GNN`
- Edge index in COO format (`[2, E]`)
- Semi-supervised learning: train on 4 labeled nodes, evaluate on 4 held-out nodes
