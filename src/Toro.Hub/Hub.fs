namespace Toro.Hub

open System
open System.IO
open FsHttp
open Toro

/// Download files from the Hugging Face Hub.
module Hub =

    let private cacheDir =
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "toro", "hub")

    let private buildRequest (url: string) =
        match
            Environment.GetEnvironmentVariable "HF_TOKEN"
            |> Option.ofObj
        with
        | Some token ->
            http {
                GET url
                Authorization $"Bearer {token}"
            }
        | None -> http { GET url }

    /// Download a single file from Hugging Face Hub.
    /// Return the local cache path.
    let download (repoId: string) (filename: string) : Async<Result<string, ToroError>> =
        async {
            let localDir =
                Path.Combine(cacheDir, repoId.Replace("/", Path.DirectorySeparatorChar.ToString()))

            let localPath = Path.Combine(localDir, filename)

            if File.Exists localPath then
                return Ok localPath
            else
                try
                    Directory.CreateDirectory localDir |> ignore
                    let url = $"https://huggingface.co/{repoId}/resolve/main/{filename}"
                    let! response = buildRequest url |> Request.sendAsync
                    let statusCode = int response.statusCode

                    if statusCode >= 200 && statusCode < 300 then
                        let! stream = response |> Response.toStreamAsync
                        use fs = new FileStream(localPath, FileMode.Create, FileAccess.Write)
                        do! stream.CopyToAsync(fs) |> Async.AwaitTask
                        return Ok localPath
                    elif statusCode = 401 then
                        return Error(Msg "Unauthorized: set HF_TOKEN for gated models")
                    elif statusCode = 404 then
                        return Error(TensorNotFound $"{repoId}/{filename}")
                    else
                        let! body = response |> Response.toTextAsync
                        return Error(Msg $"HTTP {statusCode}: {body}")
                with ex ->
                    return Error(TorchSharpError ex)
        }

    /// Download a .safetensors file from Hugging Face Hub and load it as a tensor dictionary.
    let loadSafeTensors (repoId: string) (filename: string) : Async<Result<Map<string, Tensor>, ToroError>> =
        async {
            let! pathResult = download repoId filename

            return
                match pathResult with
                | Ok path -> SafeTensors.load path
                | Error e -> Error e
        }
