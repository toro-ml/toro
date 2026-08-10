namespace Toro.NN

open System.IO
open System.Text.Json
open Toro

/// Save and restore full training state (model + optimizer + epoch).
module Checkpoint =

    [<CLIMutable>]
    type CheckpointMeta = { Epoch: int; LearningRate: float }

    /// Save model parameters, optimizer state, and epoch number to a directory.
    let save (model: 'T) (ops: OptimizerOps) (epoch: int) (dirPath: string) : Result<unit, ToroError> =
        result {
            do! ToroError.wrap (fun () -> Directory.CreateDirectory dirPath |> ignore)
            do! Model.save model (Path.Combine(dirPath, "model.safetensors"))
            do! ops.SaveState dirPath

            let meta = {
                Epoch = epoch
                LearningRate = ops.LearningRate()
            }

            do!
                ToroError.wrap (fun () ->
                    let json =
                        JsonSerializer.Serialize(meta, JsonSerializerOptions(WriteIndented = true))

                    File.WriteAllText(Path.Combine(dirPath, "meta.json"), json))
        }

    /// Load model parameters, optimizer state, and epoch number from a directory.
    /// Return the restored epoch number.
    let load (model: 'T) (ops: OptimizerOps) (dirPath: string) : Result<int, ToroError> =
        result {
            let! _report = Model.loadInto model (Path.Combine(dirPath, "model.safetensors")) Strict
            do! ops.LoadState dirPath

            let metaPath = Path.Combine(dirPath, "meta.json")

            let! meta =
                ToroError.wrap (fun () ->
                    let json = File.ReadAllText metaPath
                    JsonSerializer.Deserialize<CheckpointMeta>(json))

            ops.SetLearningRate meta.LearningRate
            return meta.Epoch
        }
