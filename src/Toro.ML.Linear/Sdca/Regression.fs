namespace Toro.ML.Linear.Sdca

open System
open System.IO
open Microsoft.ML
open Microsoft.ML.Data
open Microsoft.ML.Trainers
open TorchSharp
open Toro
open Toro.ML
open Toro.ML.Interop

/// Common SDCA regression settings.
type RegressionConfig = {
    /// L1 regularization weight, or `None` to use the ML.NET default.
    L1Regularization: float32 option
    /// L2 regularization weight, or `None` to use the ML.NET default.
    L2Regularization: float32 option
    /// Maximum training iterations, or `None` to use the ML.NET default.
    MaximumNumberOfIterations: int option
    /// ML.NET context seed. The default is `Some 0` for reproducibility.
    Seed: int option
}

/// Constructors for SDCA regression settings.
module RegressionConfig =

    /// Create settings from the ML.NET SDCA defaults with a deterministic context seed.
    let create () : RegressionConfig =
        let options = SdcaRegressionTrainer.Options()

        {
            L1Regularization = Option.ofNullable options.L1Regularization
            L2Regularization = Option.ofNullable options.L2Regularization
            MaximumNumberOfIterations = Option.ofNullable options.MaximumNumberOfIterations
            Seed = Some 0
        }

/// A fitted SDCA regression model.
[<Sealed>]
type RegressionModel internal (featureCount: int, transformer: RegressionPredictionTransformer<LinearRegressionModelParameters>)
    =

    /// Number of input features expected by the model.
    member _.FeatureCount = featureCount

    /// Underlying ML.NET model.
    member _.Transformer = transformer

/// SDCA regression operations.
module Regression =

    let private validateConfig (config: RegressionConfig) =
        match config.L1Regularization with
        | Some value when
            value < 0.0f
            || Single.IsNaN value
            || Single.IsInfinity value
            ->
            invalidArg (nameof config) "L1Regularization must be finite and non-negative."
        | _ -> ()

        match config.L2Regularization with
        | Some value when
            value < 0.0f
            || Single.IsNaN value
            || Single.IsInfinity value
            ->
            invalidArg (nameof config) "L2Regularization must be finite and non-negative."
        | _ -> ()

        match config.MaximumNumberOfIterations with
        | Some value when value <= 0 -> invalidArg (nameof config) "MaximumNumberOfIterations must be positive."
        | _ -> ()

    let private context (seed: int option) =
        match seed with
        | Some value -> MLContext(seed = value)
        | None -> MLContext()

    let private prepareOptions (configure: SdcaRegressionTrainer.Options -> unit) : SdcaRegressionTrainer.Options =
        let options = SdcaRegressionTrainer.Options()
        configure options
        options.LabelColumnName <- Columns.Label
        options.FeatureColumnName <- Columns.Features
        options

    let private fitOptions
        (seed: int option)
        (options: SdcaRegressionTrainer.Options)
        (dataset: RegressionDataset)
        : RegressionModel =
        let mlContext = context seed
        let data = RegressionDataView.training mlContext dataset
        let transformer = mlContext.Regression.Trainers.Sdca(options).Fit data
        RegressionModel(dataset.FeatureCount, transformer)

    /// Fit an SDCA regressor with common F# settings.
    let fit (config: RegressionConfig) (dataset: RegressionDataset) : RegressionModel =
        validateConfig config

        prepareOptions (fun options ->
            options.L1Regularization <- Option.toNullable config.L1Regularization
            options.L2Regularization <- Option.toNullable config.L2Regularization
            options.MaximumNumberOfIterations <- Option.toNullable config.MaximumNumberOfIterations)
        |> fun options -> fitOptions config.Seed options dataset

    /// Fit an SDCA regressor after configuring the complete ML.NET options object.
    let fitWithOptions
        (seed: int option)
        (configure: SdcaRegressionTrainer.Options -> unit)
        (dataset: RegressionDataset)
        : RegressionModel =
        prepareOptions configure
        |> fun options -> fitOptions seed options dataset

    /// Predict target values. Values are returned as a detached CPU float32 tensor.
    let predict (features: Tensor) (model: RegressionModel) : Tensor =
        let mlContext = MLContext()
        let featureCount, data = TensorDataView.scoring mlContext features

        if featureCount <> model.FeatureCount then
            invalidArg (nameof features) $"Features have width {featureCount}, expected {model.FeatureCount}."

        model.Transformer.Transform(data)
        |> TensorDataView.float32Column Columns.Score

    /// Evaluate a fitted regressor with ML.NET regression metrics.
    let evaluate (dataset: RegressionDataset) (model: RegressionModel) : RegressionMetrics =
        if dataset.FeatureCount <> model.FeatureCount then
            invalidArg (nameof dataset) $"Dataset features have width {dataset.FeatureCount}, expected {model.FeatureCount}."

        let mlContext = MLContext()
        let data = RegressionDataView.training mlContext dataset
        let scored = model.Transformer.Transform data
        mlContext.Regression.Evaluate(scored, Columns.Label, Columns.Score)

    /// Save a fitted model in the ML.NET zip format.
    let save (path: string) (model: RegressionModel) : unit =
        let mlContext = MLContext()
        let inputSchema = TensorDataView.schema mlContext model.FeatureCount
        mlContext.Model.Save(model.Transformer, inputSchema, path)

    /// Load a linear regression model compatible with the SDCA output type.
    let load (path: string) : RegressionModel =
        let mlContext = MLContext()
        let mutable inputSchema = Unchecked.defaultof<DataViewSchema>
        let transformer = mlContext.Model.Load(path, &inputSchema)

        match transformer with
        | :? RegressionPredictionTransformer<LinearRegressionModelParameters> as regressor ->
            if regressor.FeatureColumnName <> Columns.Features then
                raise (
                    InvalidDataException
                        $"Model feature column is '{regressor.FeatureColumnName}', expected '{Columns.Features}'."
                )

            let featureCount =
                match regressor.FeatureColumnType with
                | :? VectorDataViewType as featureType -> featureType.Size
                | featureType -> raise (InvalidDataException $"Model feature column has unsupported type '{featureType}'.")

            if featureCount <= 0 then
                raise (InvalidDataException "Model does not declare a fixed positive feature width.")

            let inputFeature =
                inputSchema
                |> Seq.tryFind (fun column -> column.Name = Columns.Features)
                |> Option.defaultWith (fun () ->
                    raise (InvalidDataException $"Model input schema does not contain the '{Columns.Features}' column."))

            match inputFeature.Type with
            | :? VectorDataViewType as inputType when
                inputType.ItemType = NumberDataViewType.Single
                && inputType.Size = featureCount
                ->
                ()
            | inputType ->
                raise (
                    InvalidDataException
                        $"Model input schema has incompatible feature type '{inputType}', expected Vector<Single>[{featureCount}]."
                )

            RegressionModel(featureCount, regressor)
        | _ -> raise (InvalidDataException "The file does not contain a compatible linear regression model.")
