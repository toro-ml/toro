namespace Toro.GNN

open Toro

module GraphUtils =
    /// Add self-loops to edge index. Returns (newEdgeIndex, numNodes).
    let addSelfLoops (edgeIndex: Tensor) (numNodes: int) : Result<Tensor, ToroError> =
        result {
            let device = edgeIndex.Device
            let! selfLoop = Tensor.arange (float numNodes, I64, device)
            let! selfLoop = selfLoop.unsqueeze 0
            let! selfLoop = selfLoop.expand [ 2; numNodes ]
            let! selfLoop = selfLoop.contiguous ()
            return! Tensor.cat ([ edgeIndex; selfLoop ], 1)
        }

    /// Compute node degree from an index vector (one row of edge_index).
    let degree (index: Tensor) (numNodes: int) (dtype: DType) (device: Device) : Result<Tensor, ToroError> =
        result {
            let numEdges = index.Shape[0]
            let! ones = Tensor.ones ([ numEdges ], dtype, device)
            let! out = Tensor.zeros ([ numNodes ], dtype, device)
            return! out.scatterAdd (0, index, ones)
        }
