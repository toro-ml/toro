namespace Toro.NN

open System
open System.IO
open System.Text.Json
open Toro

/// Save and restore versioned training state (model + optimizer + epoch).
module Checkpoint =

    let private formatVersion = 2
    let private modelFileName = "model.safetensors"
    let private optimizerFileName = "optimizer.safetensors"
    let private manifestFileName = "meta.json"

    [<CLIMutable>]
    type CheckpointMeta = {
        FormatVersion: int
        Epoch: int
        LearningRate: float
        OptimizerKind: string
    }

    let private manifestPath dirPath = Path.Combine(dirPath, manifestFileName)
    let private modelPath dirPath = Path.Combine(dirPath, modelFileName)

    let private optimizerPath dirPath =
        Path.Combine(dirPath, optimizerFileName)

    let private readAndValidateManifest (optimizer: IOptimizer) dirPath =
        let json = File.ReadAllText(manifestPath dirPath)
        let manifest = JsonSerializer.Deserialize<CheckpointMeta>(json)

        if obj.ReferenceEquals(manifest, null) then
            invalidOp "Checkpoint manifest is empty."

        if manifest.FormatVersion <> formatVersion then
            invalidOp
                $"Unsupported checkpoint format version {manifest.FormatVersion}; expected {formatVersion}. Version 1 and unversioned checkpoints are not supported."

        if not (String.Equals(manifest.OptimizerKind, optimizer.OptimizerKind, StringComparison.Ordinal)) then
            invalidOp $"Checkpoint optimizer kind is '{manifest.OptimizerKind}', but '{optimizer.OptimizerKind}' was provided."

        if manifest.Epoch < 0 then
            invalidOp $"Checkpoint epoch must be non-negative, but is {manifest.Epoch}."

        if
            Double.IsNaN manifest.LearningRate
            || Double.IsInfinity manifest.LearningRate
            || manifest.LearningRate < 0.0
        then
            invalidOp $"Checkpoint learning rate must be finite and non-negative, but is {manifest.LearningRate}."

        manifest

    /// Save canonical model state, optimizer state, and a Version 2 manifest.
    let save (model: 'T) (optimizer: IOptimizer) (epoch: int) (dirPath: string) : unit =
        if epoch < 0 then
            invalidArg (nameof epoch) "Checkpoint epoch must be non-negative."

        Directory.CreateDirectory dirPath |> ignore
        Model.save model (modelPath dirPath)
        optimizer.saveState (optimizerPath dirPath)

        let manifest = {
            FormatVersion = formatVersion
            Epoch = epoch
            LearningRate = optimizer.learningRate ()
            OptimizerKind = optimizer.OptimizerKind
        }

        let json =
            JsonSerializer.Serialize(manifest, JsonSerializerOptions(WriteIndented = true))

        File.WriteAllText(manifestPath dirPath, json)

    /// Validate a Version 2 checkpoint completely, then commit model state,
    /// optimizer state, and learning rate in that order. Return the restored epoch.
    let load (model: 'T) (optimizer: IOptimizer) (dirPath: string) : int =
        let manifest = readAndValidateManifest optimizer dirPath

        scoped {
            let modelTensors = SafeTensors.load (modelPath dirPath)
            let optimizerTensors = SafeTensors.load (optimizerPath dirPath)
            let _, commitModel = Model.prepareLoadFromDict model modelTensors None Strict

            optimizer.validateStateDict optimizerTensors

            commitModel ()
            optimizer.loadStateDict optimizerTensors
            optimizer.setLearningRate manifest.LearningRate
        }

        manifest.Epoch
