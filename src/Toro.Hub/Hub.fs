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
    let download (repoId: string) (filename: string) : Async<string> =
        async {
            let localDir =
                Path.Combine(cacheDir, repoId.Replace("/", Path.DirectorySeparatorChar.ToString()))

            let localPath = Path.Combine(localDir, filename)

            if File.Exists localPath then
                return localPath
            else
                Directory.CreateDirectory localDir |> ignore
                let url = $"https://huggingface.co/{repoId}/resolve/main/{filename}"
                let! response = buildRequest url |> Request.sendAsync
                let statusCode = int response.statusCode

                if statusCode >= 200 && statusCode < 300 then
                    let! stream = response |> Response.toStreamAsync
                    use fs = new FileStream(localPath, FileMode.Create, FileAccess.Write)
                    do! stream.CopyToAsync(fs) |> Async.AwaitTask
                    return localPath
                elif statusCode = 401 then
                    return failwith "Unauthorized: set HF_TOKEN for gated models"
                elif statusCode = 404 then
                    return failwith $"Tensor not found: {repoId}/{filename}"
                else
                    let! body = response |> Response.toTextAsync
                    return failwith $"HTTP {statusCode}: {body}"
        }

    /// Download a .safetensors file from Hugging Face Hub and load it as a tensor dictionary.
    let loadSafeTensors (repoId: string) (filename: string) : Async<Map<string, Tensor>> =
        async {
            let! path = download repoId filename
            return SafeTensors.load path
        }
