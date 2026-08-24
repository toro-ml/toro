namespace Toro.ML.Interop

open System
open System.Collections.Generic
open System.ComponentModel
open Microsoft.ML
open Microsoft.ML.Data
open TorchSharp
open Toro
open Toro.ML

/// Fixed ML.NET column names used by Toro.ML algorithm packages.
[<EditorBrowsable(EditorBrowsableState.Never)>]
module Columns =

    [<Literal>]
    let Features = "Features"

    [<Literal>]
    let Label = "Label"

    [<Literal>]
    let GroupId = "GroupId"

    [<Literal>]
    let Score = "Score"

/// Internal ML.NET training row exposed for algorithm extension packages.
[<CLIMutable; EditorBrowsable(EditorBrowsableState.Never)>]
type RankingRow = {
    /// Fixed-width feature vector.
    Features: float32 array
    /// Relevance label.
    Label: float32
    /// One-based ML.NET query-group key.
    [<KeyType(4294967295UL)>]
    GroupId: uint32
}

/// Internal ML.NET scoring row exposed for algorithm extension packages.
[<CLIMutable; EditorBrowsable(EditorBrowsableState.Never)>]
type ScoringRow = {
    /// Fixed-width feature vector.
    Features: float32 array
}

/// Tensor-to-IDataView conversion for algorithm extension packages.
[<EditorBrowsable(EditorBrowsableState.Never)>]
module RankingDataView =

    let private copyFloat32 (tensor: Tensor) : float32 array =
        scopedExplicit {
            let cpu = tensor.cpu ()
            let contiguous = cpu.contiguous ()
            return contiguous.data<float32>().ToArray()
        }

    let private copyInt64 (tensor: Tensor) : int64 array =
        scopedExplicit {
            let cpu = tensor.cpu ()
            let contiguous = cpu.contiguous ()
            return contiguous.data<int64>().ToArray()
        }

    let private encodeGroups (groupIds: Tensor) : uint32 array =
        let values = copyInt64 groupIds
        let encoded = Array.zeroCreate<uint32> values.Length
        let seen = HashSet<int64>()
        let mutable current = values[0]
        // LoadFromEnumerable offsets raw key values by one, since zero is the
        // missing-value representation in an IDataView key column.
        let mutable key = 0u
        seen.Add current |> ignore
        encoded[0] <- key

        for i in 1 .. values.Length - 1 do
            let value = values[i]

            if value <> current then
                if not (seen.Add value) then
                    invalidArg (nameof groupIds) $"Group id {value} appears in more than one non-contiguous block."

                if key = UInt32.MaxValue - 1u then
                    invalidArg (nameof groupIds) "The dataset contains too many query groups."

                key <- key + 1u
                current <- value

            encoded[i] <- key

        encoded

    let private setFeatureType (width: int) (schema: SchemaDefinition) =
        schema
        |> Seq.iter (fun column ->
            if column.ColumnName = Columns.Features then
                column.ColumnType <- VectorDataViewType(NumberDataViewType.Single, width))

    /// Convert a validated ranking dataset to an ML.NET training view.
    let training (mlContext: MLContext) (dataset: RankingDataset) : IDataView =
        RankingDataset.validate dataset
        let width = dataset.FeatureCount
        let rowCount = int dataset.RowCount
        let flat = copyFloat32 dataset.Features
        let labels = copyFloat32 dataset.Labels
        let groups = encodeGroups dataset.GroupIds

        let rows =
            Array.init rowCount (fun rowIndex ->
                let features = Array.zeroCreate<float32> width
                Array.Copy(flat, rowIndex * width, features, 0, width)

                {
                    Features = features
                    Label = labels[rowIndex]
                    GroupId = groups[rowIndex]
                })

        let schema = SchemaDefinition.Create(typeof<RankingRow>)
        setFeatureType width schema
        mlContext.Data.LoadFromEnumerable(rows, schema)

    /// Convert a feature matrix to an ML.NET scoring view and return its width.
    let scoring (mlContext: MLContext) (features: Tensor) : int * IDataView =
        RankingDataset.validateFeatures (nameof features) features
        let width = int features.shape[1]
        let rowCount = int features.shape[0]
        let flat = copyFloat32 features

        let rows =
            Array.init rowCount (fun rowIndex ->
                let values = Array.zeroCreate<float32> width
                Array.Copy(flat, rowIndex * width, values, 0, width)
                { Features = values })

        let schema = SchemaDefinition.Create(typeof<ScoringRow>)
        setFeatureType width schema
        width, mlContext.Data.LoadFromEnumerable(rows, schema)

    /// Copy a scalar float32 IDataView column to a detached CPU tensor.
    let float32Column (columnName: string) (data: IDataView) : Tensor =
        data.GetColumn<float32>(columnName)
        |> Seq.toArray
        |> fun values -> torch.tensor (values, dtype = torch.float32, device = torch.CPU)

    /// Create the ML.NET input schema for a feature width.
    let schema (mlContext: MLContext) (featureCount: int) : DataViewSchema =
        if featureCount <= 0 then
            invalidArg (nameof featureCount) "Feature count must be positive."

        let definition = SchemaDefinition.Create(typeof<ScoringRow>)
        setFeatureType featureCount definition

        mlContext.Data.LoadFromEnumerable(Array.empty<ScoringRow>, definition)
        |> fun data -> data.Schema
