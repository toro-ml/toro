open System
open System.IO

let root = Path.GetFullPath(Path.Combine(__SOURCE_DIRECTORY__, ".."))

let exec (cmd: string) (args: string) =
    let psi = Diagnostics.ProcessStartInfo(cmd, args)
    psi.WorkingDirectory <- root
    psi.RedirectStandardOutput <- true
    psi.RedirectStandardError <- true
    psi.UseShellExecute <- false
    let p = Diagnostics.Process.Start(psi)
    let stdout = p.StandardOutput.ReadToEndAsync()
    let stderr = p.StandardError.ReadToEndAsync()
    p.WaitForExit()
    (p.ExitCode, stdout.Result, stderr.Result)

let requireSuccess description (code, stdout: string, stderr: string) =
    if code <> 0 then
        eprintfn "%s failed with exit code %d." description code

        if not (String.IsNullOrWhiteSpace stdout) then
            eprintfn "stdout:\n%s" (stdout.TrimEnd())

        if not (String.IsNullOrWhiteSpace stderr) then
            eprintfn "stderr:\n%s" (stderr.TrimEnd())

        exit code

    stdout

let copyDeps () =
    let libDir = Path.Combine(root, ".fsdocs-libs")

    let projects = [ "src/Toro.Text"; "src/Toro.Vision" ]

    for proj in projects do
        exec "dotnet" $"publish {proj} -c Release --no-build -o {libDir}"
        |> requireSuccess $"publish {proj}"
        |> ignore

    let targets = [
        ("src/Toro.Text/bin/Release/net10.0", [ "Microsoft.ML.Tokenizers.dll" ])
        ("src/Toro.Vision/bin/Release/net10.0", [ "TorchVision.dll"; "SkiaSharp.dll" ])
    ]

    for (destRel, dlls) in targets do
        let dest = Path.Combine(root, destRel)

        for dll in dlls do
            let src = Path.Combine(libDir, dll)

            if File.Exists(src) then
                File.Copy(src, Path.Combine(dest, dll), true)

let run () =
    let stdout =
        exec "dotnet" "fsdocs build --clean --input .fsdocs-input --output .fsdocs-out --properties Configuration=Release"
        |> requireSuccess "fsdocs"

    printfn "%s" stdout

Directory.CreateDirectory(Path.Combine(root, ".fsdocs-input"))
|> ignore

copyDeps ()
run ()
printfn "FSDocs HTML generated to .fsdocs-out/"
