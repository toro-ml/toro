open System
open System.IO

let root = Path.GetFullPath(Path.Combine(__SOURCE_DIRECTORY__, ".."))

let run () =
    let psi =
        Diagnostics.ProcessStartInfo(
            "dotnet",
            "fsdocs build --clean --input .fsdocs-input --output .fsdocs-out --properties Configuration=Release"
        )

    psi.WorkingDirectory <- root
    psi.RedirectStandardOutput <- true
    psi.RedirectStandardError <- true
    psi.UseShellExecute <- false
    let p = Diagnostics.Process.Start(psi)
    p.WaitForExit()

    if p.ExitCode <> 0 then
        eprintfn "fsdocs failed: %s" (p.StandardError.ReadToEnd())
        exit 1

    printfn "%s" (p.StandardOutput.ReadToEnd())

Directory.CreateDirectory(Path.Combine(root, ".fsdocs-input"))
|> ignore

run ()
printfn "FSDocs HTML generated to .fsdocs-out/"
