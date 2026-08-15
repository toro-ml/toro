namespace Toro.Hub

open System
open System.IO
open System.Security.Cryptography
open System.Text
open FsHttp

/// Reference to one file at an explicit Hugging Face Hub repository revision.
type HubFile = {
    RepoId: string
    Revision: string
    Filename: string
}

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

    let private pathSegments argumentName (value: string) =
        if String.IsNullOrWhiteSpace value then
            invalidArg argumentName "Hub path must not be empty."

        if value.Contains('\\') then
            invalidArg argumentName $"Hub paths must use '/' separators: '{value}'."

        let segments = value.Split('/')

        if
            segments.Length = 0
            || segments
               |> Array.exists (fun part ->
                   String.IsNullOrEmpty part
                   || part = "."
                   || part = ".."
                   || part.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0)
        then
            invalidArg argumentName $"Invalid Hub path: '{value}'."

        segments

    let private urlPath argumentName value =
        pathSegments argumentName value
        |> Array.map Uri.EscapeDataString
        |> String.concat "/"

    let private revisionCacheKey (revision: string) =
        if String.IsNullOrWhiteSpace revision then
            invalidArg (nameof revision) "Hub revision must not be empty."

        revision
        |> Encoding.UTF8.GetBytes
        |> SHA256.HashData
        |> Convert.ToHexString
        |> fun value -> value.ToLowerInvariant()

    let private localPath (file: HubFile) =
        let repoSegments = pathSegments (nameof file.RepoId) file.RepoId
        let filenameSegments = pathSegments (nameof file.Filename) file.Filename

        Array.concat [
            [| cacheDir |]
            repoSegments
            [| revisionCacheKey file.Revision |]
            filenameSegments
        ]
        |> Path.Combine

    let private url (file: HubFile) =
        let repoId = urlPath (nameof file.RepoId) file.RepoId
        let revision = Uri.EscapeDataString file.Revision
        let filename = urlPath (nameof file.Filename) file.Filename
        $"https://huggingface.co/{repoId}/resolve/{revision}/{filename}"

    /// Download a single file from Hugging Face Hub.
    /// Return the local cache path.
    let download (file: HubFile) : Async<string> =
        async {
            let localPath = localPath file

            if File.Exists localPath then
                return localPath
            else
                let localDir = Path.GetDirectoryName localPath
                Directory.CreateDirectory localDir |> ignore
                let! response = buildRequest (url file) |> Request.sendAsync
                let statusCode = int response.statusCode

                if statusCode >= 200 && statusCode < 300 then
                    let temporaryPath = $"{localPath}.{Guid.NewGuid():N}.part"

                    try
                        use! stream = response |> Response.toStreamAsync

                        use fs =
                            new FileStream(temporaryPath, FileMode.CreateNew, FileAccess.Write, FileShare.None)

                        do! stream.CopyToAsync(fs) |> Async.AwaitTask
                        fs.Flush(true)

                        try
                            File.Move(temporaryPath, localPath)
                        with :? IOException when File.Exists localPath ->
                            File.Delete temporaryPath

                    finally
                        if File.Exists temporaryPath then
                            File.Delete temporaryPath

                    return localPath
                elif statusCode = 401 then
                    return failwith "Unauthorized: set HF_TOKEN for gated models"
                elif statusCode = 404 then
                    return failwith $"Hub file not found: {file.RepoId}@{file.Revision}/{file.Filename}"
                else
                    let! body = response |> Response.toTextAsync
                    return failwith $"HTTP {statusCode}: {body}"
        }
