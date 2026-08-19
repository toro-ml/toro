namespace Toro.Models

open System
open System.IO
open Toro

module internal LocalModelAssets =

    let openReader (label: string) (directory: string) =
        if String.IsNullOrWhiteSpace directory then
            invalidArg (nameof directory) "Model directory must not be empty."

        let directory = Path.GetFullPath directory

        if not (Directory.Exists directory) then
            invalidArg (nameof directory) $"Model directory does not exist: '{directory}'."

        let configPath = Path.Combine(directory, "config.json")

        if not (File.Exists configPath) then
            invalidOp $"{label} config is missing: '{configPath}'."

        let singlePath = Path.Combine(directory, "model.safetensors")
        let indexPath = Path.Combine(directory, "model.safetensors.index.json")

        let reader =
            match File.Exists singlePath, File.Exists indexPath with
            | true, false -> SafeTensors.openFile singlePath
            | false, true -> SafeTensors.openIndex indexPath
            | true, true -> invalidOp "Model directory contains both single-file and sharded SafeTensors state."
            | false, false -> invalidOp "Model directory contains neither model.safetensors nor model.safetensors.index.json."

        configPath, reader

    let dtype (label: string) (tensorName: string) (reader: SafeTensorReader) =
        reader.Metadata
        |> Map.tryFind tensorName
        |> Option.map _.DType
        |> Option.defaultWith (fun () -> invalidOp $"{label} state is missing '{tensorName}'.")
