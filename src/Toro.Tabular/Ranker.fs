namespace Toro.Tabular

open System
open Microsoft.ML
open Microsoft.ML.Data
open Microsoft.ML.Trainers.FastTree
open TorchSharp
open Toro

/// Hyperparameters for FastTree ranking.
type RankerConfig = {
    NumberOfTrees: int
    NumberOfLeaves: int
    MinimumExampleCountPerLeaf: int
    Seed: int option
}

/// Constructors for FastTree ranking hyperparameters.
module RankerConfig =

    /// Create a small FastTree ranking configuration (8 trees, 4 leaves).
    let create () : RankerConfig = {
        NumberOfTrees = 8
        NumberOfLeaves = 4
        MinimumExampleCountPerLeaf = 1
        Seed = Some 0
    }

/// A fitted FastTree ranker. `Transformer` is the underlying ML.NET model.
type Ranker = {
    FeatureNames: string list
    Group: string
    FeatureCount: int
    Transformer: ITransformer
}

/// ML.NET row type used to load ranking tables. Not part of the table API.
[<CLIMutable>]
[<System.ComponentModel.EditorBrowsable(System.ComponentModel.EditorBrowsableState.Never)>]
type RankingRow = {
    Features: float32[]
    Label: float32
    [<KeyType(65536UL)>]
    GroupId: uint32
}

module private RankingData =
    let float32Array (name: string) (table: Table) : float32 array =
        match Table.column name table with
        | Floats tensor
        | Ints tensor -> tensor.to_type(torch.float32).contiguous().cpu().data<float32>().ToArray()
        | Vectors _ -> invalidArg (nameof name) $"Column '{name}' must be a 1-d numeric column."

    let groupIds (name: string) (table: Table) : uint32 array =
        float32Array name table
        |> Array.map (fun value ->
            if value < 0.0f || value > float32 UInt32.MaxValue then
                invalidArg (nameof name) $"Column '{name}' has a group id outside the UInt32 range."

            uint32 value)

    let dataView
        (mlContext: MLContext)
        (features: string list)
        (label: string option)
        (group: string)
        (table: Table)
        : int * IDataView =
        let matrix = Table.features features table
        let n = int table.Length
        let width = int matrix.shape[1]

        let flat =
            matrix.to_type(torch.float32).contiguous().cpu().data<float32>().ToArray()

        let labels =
            match label with
            | Some name -> float32Array name table
            | None -> Array.zeroCreate n

        let groups = groupIds group table

        let rows =
            Array.init n (fun i ->
                let featureRow = Array.zeroCreate width
                Array.Copy(flat, i * width, featureRow, 0, width)

                {
                    Features = featureRow
                    Label = labels[i]
                    GroupId = groups[i]
                })

        let schema = SchemaDefinition.Create(typeof<RankingRow>)

        schema
        |> Seq.iter (fun item ->
            if item.ColumnName = "Features" then
                item.ColumnType <- VectorDataViewType(NumberDataViewType.Single, width))

        width, mlContext.Data.LoadFromEnumerable(rows, schema)

/// FastTree ranking over named table columns.
module Ranker =

    /// Fit a FastTree ranker. `group` is the query-id column used as the ML.NET row group.
    let fitWith (config: RankerConfig) (features: string list) (label: string) (group: string) (table: Table) : Ranker =
        if config.NumberOfTrees <= 0 then
            invalidArg (nameof config) "NumberOfTrees must be positive."

        if config.NumberOfLeaves <= 1 then
            invalidArg (nameof config) "NumberOfLeaves must be greater than 1."

        if config.MinimumExampleCountPerLeaf <= 0 then
            invalidArg (nameof config) "MinimumExampleCountPerLeaf must be positive."

        let mlContext =
            match config.Seed with
            | Some seed -> MLContext(seed = seed)
            | None -> MLContext()

        let featureCount, data =
            RankingData.dataView mlContext features (Some label) group table

        let options = FastTreeRankingTrainer.Options()
        options.NumberOfTrees <- config.NumberOfTrees
        options.NumberOfLeaves <- config.NumberOfLeaves
        options.MinimumExampleCountPerLeaf <- config.MinimumExampleCountPerLeaf
        options.LabelColumnName <- "Label"
        options.FeatureColumnName <- "Features"
        options.RowGroupColumnName <- "GroupId"

        let pipeline = mlContext.Ranking.Trainers.FastTree options

        {
            FeatureNames = features
            Group = group
            FeatureCount = featureCount
            Transformer = pipeline.Fit data
        }

    /// Fit a FastTree ranker with `RankerConfig.create` defaults.
    let fit (features: string list) (label: string) (group: string) (table: Table) : Ranker =
        fitWith (RankerConfig.create ()) features label group table

    /// Score each row. The table must contain the fitted feature columns and group column.
    let predict (table: Table) (ranker: Ranker) : Tensor =
        let mlContext = MLContext()

        let featureCount, data =
            RankingData.dataView mlContext ranker.FeatureNames None ranker.Group table

        if featureCount <> ranker.FeatureCount then
            invalidArg (nameof table) $"Expected {ranker.FeatureCount} feature columns, but the table has {featureCount}."

        let scored = ranker.Transformer.Transform data

        scored.GetColumn<float32>("Score")
        |> Seq.toArray
        |> fun values -> torch.tensor (values, dtype = torch.float32, device = torch.CPU)
