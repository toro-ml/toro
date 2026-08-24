namespace Toro.ML

open TorchSharp
open Toro

/// A borrowed tensor dataset for regression tasks.
[<Sealed>]
type RegressionDataset internal (features: Tensor, labels: Tensor) =

    /// Feature matrix with shape `[n, f]` and dtype `float32`.
    member _.Features = features

    /// Target values with shape `[n]` and dtype `float32`.
    member _.Labels = labels

    /// Number of examples.
    member _.RowCount = features.shape[0]

    /// Number of features per example.
    member _.FeatureCount = int features.shape[1]

/// Constructors for regression datasets.
module RegressionDataset =

    let internal validate (dataset: RegressionDataset) =
        DatasetValidation.features (nameof dataset.Features) dataset.Features
        DatasetValidation.float32Labels (nameof dataset.Labels) dataset.Features.shape[0] dataset.Labels

    /// Create a dataset without copying or taking ownership of its tensors.
    let create (features: Tensor) (labels: Tensor) : RegressionDataset =
        let dataset = RegressionDataset(features, labels)
        validate dataset
        dataset
