namespace Toro.ML

open TorchSharp
open Toro

/// A borrowed tensor dataset for learning-to-rank tasks.
[<Sealed>]
type RankingDataset internal (features: Tensor, labels: Tensor, groupIds: Tensor) =

    /// Feature matrix with shape `[n, f]` and dtype `float32`.
    member _.Features = features

    /// Relevance labels with shape `[n]` and dtype `float32`.
    member _.Labels = labels

    /// Query identifiers with shape `[n]` and dtype `int64`.
    member _.GroupIds = groupIds

    /// Number of examples.
    member _.RowCount = features.shape[0]

    /// Number of features per example.
    member _.FeatureCount = int features.shape[1]

/// Constructors for learning-to-rank datasets.
module RankingDataset =

    let private copyGroupIds (groupIds: Tensor) : int64 array =
        scopedExplicit {
            let cpu = groupIds.cpu ()
            let contiguous = cpu.contiguous ()
            return contiguous.data<int64>().ToArray()
        }

    let private validateContiguousGroups (groupIds: Tensor) =
        let values = copyGroupIds groupIds
        let seen = System.Collections.Generic.HashSet<int64>()
        let mutable current = values[0]
        seen.Add current |> ignore

        for i in 1 .. values.Length - 1 do
            let value = values[i]

            if value <> current then
                if not (seen.Add value) then
                    invalidArg (nameof groupIds) $"Group id {value} appears in more than one non-contiguous block."

                current <- value

    let internal validate (dataset: RankingDataset) =
        DatasetValidation.features (nameof dataset.Features) dataset.Features
        DatasetValidation.float32Labels (nameof dataset.Labels) dataset.Features.shape[0] dataset.Labels

        if dataset.GroupIds.dim () <> 1 then
            invalidArg (nameof dataset.GroupIds) $"Group ids must be 1-d, but have {dataset.GroupIds.dim ()} dimensions."

        if dataset.GroupIds.dtype <> torch.int64 then
            invalidArg (nameof dataset.GroupIds) $"Group ids must have dtype int64, but have {dataset.GroupIds.dtype}."

        let rowCount = dataset.Features.shape[0]

        if dataset.GroupIds.shape[0] <> rowCount then
            invalidArg (nameof dataset.GroupIds) $"Group ids have {dataset.GroupIds.shape[0]} rows, expected {rowCount}."

        validateContiguousGroups dataset.GroupIds

    /// Create a dataset without copying or taking ownership of its tensors.
    let create (features: Tensor) (labels: Tensor) (groupIds: Tensor) : RankingDataset =
        let dataset = RankingDataset(features, labels, groupIds)
        validate dataset
        dataset
