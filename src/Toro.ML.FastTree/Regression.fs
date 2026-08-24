namespace Toro.ML.FastTree

open System
open System.IO
open Microsoft.ML
open Microsoft.ML.Data
open Microsoft.ML.Trainers.FastTree
open TorchSharp
open Toro
open Toro.ML
open Toro.ML.Interop

/// Common FastTree regression settings.
type RegressionConfig = {
    /// Number of trees in the ensemble.
    NumberOfTrees: int
    /// Maximum number of leaves in each tree.
    NumberOfLeaves: int
    /// Minimum number of examples required in a leaf.
    MinimumExampleCountPerLeaf: int
    /// Shrinkage rate applied to each tree.
    LearningRate: float
    /// Random seed. The default is `Some 0` for reproducibility.
    Seed: int option
}

/// Constructors for FastTree regression settings.
module RegressionConfig =

    /// Create settings from the ML.NET FastTree defaults with a deterministic seed.
    let create () : RegressionConfig =
        let options = FastTreeRegressionTrainer.Options()

        {
            NumberOfTrees = options.NumberOfTrees
            NumberOfLeaves = options.NumberOfLeaves
            MinimumExampleCountPerLeaf = options.MinimumExampleCountPerLeaf
            LearningRate = options.LearningRate
            Seed = Some 0
        }

/// A fitted FastTree regression model.
[<Sealed>]
type RegressionModel
    internal (featureCount: int, transformer: RegressionPredictionTransformer<FastTreeRegressionModelParameters>) =

    /// Number of input features expected by the model.
    member _.FeatureCount = featureCount

    /// Underlying ML.NET model.
    member _.Transformer = transformer

/// FastTree regression operations.
module Regression =

    let private validateConfig (config: RegressionConfig) =
        if config.NumberOfTrees <= 0 then
            invalidArg (nameof config) "NumberOfTrees must be positive."

        if config.NumberOfLeaves <= 1 then
            invalidArg (nameof config) "NumberOfLeaves must be greater than one."

        if config.MinimumExampleCountPerLeaf <= 0 then
            invalidArg (nameof config) "MinimumExampleCountPerLeaf must be positive."

        if
            config.LearningRate <= 0.0
            || Double.IsNaN config.LearningRate
            || Double.IsInfinity config.LearningRate
        then
            invalidArg (nameof config) "LearningRate must be finite and positive."

    let private context (seed: int option) =
        match seed with
        | Some value -> MLContext(seed = value)
        | None -> MLContext()

    let private prepareOptions (configure: FastTreeRegressionTrainer.Options -> unit) : FastTreeRegressionTrainer.Options =
        let options = FastTreeRegressionTrainer.Options()
        configure options
        options.LabelColumnName <- Columns.Label
        options.FeatureColumnName <- Columns.Features
        options

    let private fitOptions
        (seed: int option)
        (options: FastTreeRegressionTrainer.Options)
        (dataset: RegressionDataset)
        : RegressionModel =
        let mlContext = context seed
        let data = RegressionDataView.training mlContext dataset
        let transformer = mlContext.Regression.Trainers.FastTree(options).Fit data
        RegressionModel(dataset.FeatureCount, transformer)

    /// Fit a FastTree regressor with common F# settings.
    let fit (config: RegressionConfig) (dataset: RegressionDataset) : RegressionModel =
        validateConfig config

        prepareOptions (fun options ->
            options.NumberOfTrees <- config.NumberOfTrees
            options.NumberOfLeaves <- config.NumberOfLeaves
            options.MinimumExampleCountPerLeaf <- config.MinimumExampleCountPerLeaf
            options.LearningRate <- config.LearningRate

            match config.Seed with
            | Some seed -> options.Seed <- seed
            | None -> ())
        |> fun options -> fitOptions config.Seed options dataset

    /// Fit a FastTree regressor after configuring the complete ML.NET options object.
    let fitWithOptions
        (seed: int option)
        (configure: FastTreeRegressionTrainer.Options -> unit)
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

    /// Load a FastTree regression model from the ML.NET zip format.
    let load (path: string) : RegressionModel =
        let mlContext = MLContext()
        let mutable inputSchema = Unchecked.defaultof<DataViewSchema>
        let transformer = mlContext.Model.Load(path, &inputSchema)

        match transformer with
        | :? RegressionPredictionTransformer<FastTreeRegressionModelParameters> as regressor ->
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
        | _ -> raise (InvalidDataException "The file does not contain a FastTree regression model.")
