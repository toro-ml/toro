namespace Toro.GNN

open Toro

/// Graph in COO format for GNN layers.
type GraphData = {
    /// Node feature matrix [N, F].
    X: Tensor
    /// Edge index in COO format [2, E]. Row 0 = source, row 1 = target.
    EdgeIndex: Tensor
    /// Optional edge features [E, D].
    EdgeAttr: Tensor option
    /// Optional batch vector [N] mapping each node to its graph index.
    Batch: Tensor option
}

module GraphData =
    /// Create a graph with node features and edge index.
    let create (x: Tensor) (edgeIndex: Tensor) : GraphData = {
        X = x
        EdgeIndex = edgeIndex
        EdgeAttr = None
        Batch = None
    }

    /// Create a graph with edge attributes.
    let createWithEdgeAttr (x: Tensor) (edgeIndex: Tensor) (edgeAttr: Tensor) : GraphData = {
        X = x
        EdgeIndex = edgeIndex
        EdgeAttr = Some edgeAttr
        Batch = None
    }

    /// Return the number of nodes.
    let numNodes (g: GraphData) = g.X.shape[0]

    /// Return the number of edges.
    let numEdges (g: GraphData) = g.EdgeIndex.shape[1]
