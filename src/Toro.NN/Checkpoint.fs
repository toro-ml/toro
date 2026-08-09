namespace Toro.NN

open System.IO
open System.Text.Json
open Toro

/// Save and restore full training state (model + optimizer + epoch).
module Checkpoint =

    [<CLIMutable>]
    type CheckpointMeta = { Epoch: int; LearningRate: float }

    /// Save model parameters, optimizer state, and epoch number to a directory.
    let save (model: 'T) (opt: IOptimizer) (epoch: int) (dirPath: string) : Result<unit, ToroError> =
        result {
            do! ToroError.wrap (fun () -> Directory.CreateDirectory dirPath |> ignore)
            do! Model.save model (Path.Combine(dirPath, "model.safetensors"))
            do! opt.saveState dirPath

            let meta = {
                Epoch = epoch
                LearningRate = opt.learningRate ()
            }

            do!
                ToroError.wrap (fun () ->
                    let json =
                        JsonSerializer.Serialize(meta, JsonSerializerOptions(WriteIndented = true))

                    File.WriteAllText(Path.Combine(dirPath, "meta.json"), json))
        }

    /// Load model parameters, optimizer state, and epoch number from a directory.
    /// Return the restored epoch number.
    let load (model: 'T) (opt: IOptimizer) (dirPath: string) : Result<int, ToroError> =
        result {
            let! _report = Model.loadInto model (Path.Combine(dirPath, "model.safetensors")) Strict
            do! opt.loadState dirPath

            let metaPath = Path.Combine(dirPath, "meta.json")

            let! meta =
                ToroError.wrap (fun () ->
                    let json = File.ReadAllText metaPath
                    JsonSerializer.Deserialize<CheckpointMeta>(json))

            opt.setLearningRate meta.LearningRate
            return meta.Epoch
        }
