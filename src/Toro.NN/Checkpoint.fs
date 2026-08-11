namespace Toro.NN

open System.IO
open System.Text.Json
open Toro

/// Save and restore full training state (model + optimizer + epoch).
module Checkpoint =

    [<CLIMutable>]
    type CheckpointMeta = { Epoch: int; LearningRate: float }

    /// Save model parameters, optimizer state, and epoch number to a directory.
    let save (model: 'T) (ops: OptimizerOps) (epoch: int) (dirPath: string) : unit =
        Directory.CreateDirectory dirPath |> ignore
        Model.save model (Path.Combine(dirPath, "model.safetensors"))
        ops.SaveState dirPath

        let meta = {
            Epoch = epoch
            LearningRate = ops.LearningRate()
        }

        let json =
            JsonSerializer.Serialize(meta, JsonSerializerOptions(WriteIndented = true))

        File.WriteAllText(Path.Combine(dirPath, "meta.json"), json)

    /// Load model parameters, optimizer state, and epoch number from a directory.
    /// Return the restored epoch number.
    let load (model: 'T) (ops: OptimizerOps) (dirPath: string) : int =
        let _report =
            Model.loadInto model (Path.Combine(dirPath, "model.safetensors")) Strict

        ops.LoadState dirPath

        let metaPath = Path.Combine(dirPath, "meta.json")

        let json = File.ReadAllText metaPath
        let meta = JsonSerializer.Deserialize<CheckpointMeta>(json)

        ops.SetLearningRate meta.LearningRate
        meta.Epoch
