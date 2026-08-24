namespace Toro.ML.LightGbm

open System
open System.IO
open Microsoft.ML
open Microsoft.ML.Data
open Microsoft.ML.Trainers.LightGbm
open TorchSharp
open Toro
open Toro.ML
open Toro.ML.Interop

/// Common LightGBM ranking settings.
type RankingConfig = {
    /// Number of boosting iterations.
    NumberOfIterations: int
    /// Maximum number of leaves, or `None` to use ML.NET auto-tuning.
    NumberOfLeaves: int option
    /// Minimum examples per leaf, or `None` to use ML.NET auto-tuning.
    MinimumExampleCountPerLeaf: int option
    /// Shrinkage rate, or `None` to use ML.NET auto-tuning.
    LearningRate: float option
    /// Random seed. The default is `Some 0` for reproducibility.
    Seed: int option
}

/// Constructors for LightGBM ranking settings.
module RankingConfig =

    /// Create settings from the ML.NET LightGBM defaults with a deterministic seed.
    let create () : RankingConfig =
        let options = LightGbmRankingTrainer.Options()

        {
            NumberOfIterations = options.NumberOfIterations
            NumberOfLeaves = Option.ofNullable options.NumberOfLeaves
            MinimumExampleCountPerLeaf = Option.ofNullable options.MinimumExampleCountPerLeaf
            LearningRate = Option.ofNullable options.LearningRate
            Seed = Some 0
        }

/// A fitted LightGBM ranking model.
[<Sealed>]
type RankingModel internal (featureCount: int, transformer: RankingPredictionTransformer<LightGbmRankingModelParameters>) =

    /// Number of input features expected by the model.
    member _.FeatureCount = featureCount

    /// Underlying ML.NET model.
    member _.Transformer = transformer

/// LightGBM learning-to-rank operations.
module Ranking =

    let private validateConfig (config: RankingConfig) =
        if config.NumberOfIterations <= 0 then
            invalidArg (nameof config) "NumberOfIterations must be positive."

        match config.NumberOfLeaves with
        | Some value when value <= 1 -> invalidArg (nameof config) "NumberOfLeaves must be greater than one."
        | _ -> ()

        match config.MinimumExampleCountPerLeaf with
        | Some value when value <= 0 -> invalidArg (nameof config) "MinimumExampleCountPerLeaf must be positive."
        | _ -> ()

        match config.LearningRate with
        | Some value when
            value <= 0.0
            || Double.IsNaN value
            || Double.IsInfinity value
            ->
            invalidArg (nameof config) "LearningRate must be finite and positive."
        | _ -> ()

    let private context (seed: int option) =
        match seed with
        | Some value -> MLContext(seed = value)
        | None -> MLContext()

    let private prepareOptions (configure: LightGbmRankingTrainer.Options -> unit) : LightGbmRankingTrainer.Options =
        let options = LightGbmRankingTrainer.Options()
        configure options
        options.LabelColumnName <- Columns.Label
        options.FeatureColumnName <- Columns.Features
        options.RowGroupColumnName <- Columns.GroupId
        options

    let private fitOptions
        (seed: int option)
        (options: LightGbmRankingTrainer.Options)
        (dataset: RankingDataset)
        : RankingModel =
        let mlContext = context seed
        let data = RankingDataView.training mlContext dataset
        let transformer = mlContext.Ranking.Trainers.LightGbm(options).Fit data
        RankingModel(dataset.FeatureCount, transformer)

    /// Fit a LightGBM ranker with common F# settings.
    let fit (config: RankingConfig) (dataset: RankingDataset) : RankingModel =
        validateConfig config

        prepareOptions (fun options ->
            options.NumberOfIterations <- config.NumberOfIterations
            options.NumberOfLeaves <- Option.toNullable config.NumberOfLeaves
            options.MinimumExampleCountPerLeaf <- Option.toNullable config.MinimumExampleCountPerLeaf
            options.LearningRate <- Option.toNullable config.LearningRate
            options.Seed <- Option.toNullable config.Seed)
        |> fun options -> fitOptions config.Seed options dataset

    /// Fit a LightGBM ranker after configuring the complete ML.NET options object.
    let fitWithOptions
        (seed: int option)
        (configure: LightGbmRankingTrainer.Options -> unit)
        (dataset: RankingDataset)
        : RankingModel =
        prepareOptions configure
        |> fun options -> fitOptions seed options dataset

    /// Score a feature matrix. Scores are returned as a detached CPU float32 tensor.
    let predict (features: Tensor) (model: RankingModel) : Tensor =
        let mlContext = MLContext()
        let featureCount, data = RankingDataView.scoring mlContext features

        if featureCount <> model.FeatureCount then
            invalidArg (nameof features) $"Features have width {featureCount}, expected {model.FeatureCount}."

        model.Transformer.Transform(data)
        |> RankingDataView.float32Column Columns.Score

    /// Evaluate a fitted ranker with ML.NET ranking metrics.
    let evaluate (dataset: RankingDataset) (model: RankingModel) : RankingMetrics =
        if dataset.FeatureCount <> model.FeatureCount then
            invalidArg (nameof dataset) $"Dataset features have width {dataset.FeatureCount}, expected {model.FeatureCount}."

        let mlContext = MLContext()
        let data = RankingDataView.training mlContext dataset
        let scored = model.Transformer.Transform data
        mlContext.Ranking.Evaluate(scored, Columns.Label, Columns.GroupId, Columns.Score)

    /// Save a fitted model in the ML.NET zip format.
    let save (path: string) (model: RankingModel) : unit =
        let mlContext = MLContext()
        let inputSchema = RankingDataView.schema mlContext model.FeatureCount
        mlContext.Model.Save(model.Transformer, inputSchema, path)

    /// Load a LightGBM ranking model from the ML.NET zip format.
    let load (path: string) : RankingModel =
        let mlContext = MLContext()
        let mutable inputSchema = Unchecked.defaultof<DataViewSchema>
        let transformer = mlContext.Model.Load(path, &inputSchema)

        match transformer with
        | :? RankingPredictionTransformer<LightGbmRankingModelParameters> as ranker ->
            if ranker.FeatureColumnName <> Columns.Features then
                raise (
                    InvalidDataException $"Model feature column is '{ranker.FeatureColumnName}', expected '{Columns.Features}'."
                )

            let featureCount =
                match ranker.FeatureColumnType with
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

            RankingModel(featureCount, ranker)
        | _ -> raise (InvalidDataException "The file does not contain a LightGBM ranking model.")
