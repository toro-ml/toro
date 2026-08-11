namespace Toro.GNN

open TorchSharp
open Toro

module GraphUtils =
    /// Add self-loops to edge index. Returns (newEdgeIndex, numNodes).
    let addSelfLoops (edgeIndex: Tensor) (numNodes: int64) : Tensor =
        let device = edgeIndex.device

        let selfLoop =
            torch.arange (scalar (float numNodes), dtype = torch.int64, device = device)

        let selfLoop = selfLoop.unsqueeze 0L
        let selfLoop = selfLoop.expand [| 2L; numNodes |]
        let selfLoop = selfLoop.contiguous ()
        torch.cat ([| edgeIndex; selfLoop |], 1L)

    /// Compute node degree from an index vector (one row of edge_index).
    let degree (index: Tensor) (numNodes: int64) (dtype: torch.ScalarType) (device: torch.Device) : Tensor =
        let numEdges = index.shape[0]
        let ones = torch.ones ([| numEdges |], dtype = dtype, device = device)
        let out = torch.zeros ([| numNodes |], dtype = dtype, device = device)
        out.scatter_add (0L, index, ones)
