module GenerationTests

open System
open System.Threading
open TorchSharp
open Toro
open Toro.Models
open Xunit

type private FakeCache = {
    mutable Length: int64
    mutable Disposed: bool
}

let private fakeModel () =
    let mutable lastCache: FakeCache option = None

    let model = {
        ContextLength = 32L
        EosTokenIds = Set.empty
        Device = torch.CPU
        CreateCache =
            fun _ _ ->
                let cache = { Length = 0L; Disposed = false }
                lastCache <- Some cache
                cache
        CacheLength = _.Length
        DisposeCache = fun cache -> cache.Disposed <- true
        Forward =
            fun input ->
                let cache = input.Cache |> Option.get
                cache.Length <- cache.Length + input.InputIds.shape[1]

                {
                    Logits = torch.zeros ([| 1L; input.InputIds.shape[1]; 4L |], dtype = torch.float32)
                    Cache = Some cache
                }
    }

    model, fun () -> lastCache |> Option.get

[<Fact>]
let ``temperature sampling is reproducible through Torch RNG`` () =
    let model, _ = fakeModel ()
    let options = GenerationOptions.temperature 0.8 8

    torch.manual_seed 123L |> ignore
    use rngState = torch.get_rng_state ()
    let first = Generation.generate options [ 1L ] model
    torch.set_rng_state rngState
    let second = Generation.generate options [ 1L ] model

    Assert.Equal<int64 list>(first, second)

[<Fact>]
let ``session observes cancellation and owns cache disposal`` () =
    let model, cache = fakeModel ()
    use cancellation = new CancellationTokenSource()

    let options = {
        GenerationOptions.greedy 2 with
            CancellationToken = cancellation.Token
    }

    let session = Generation.createSession options [ 1L ] model
    cancellation.Cancel()

    Assert.Throws<OperationCanceledException>(fun () -> session.Step() |> ignore)
    |> ignore

    (session :> System.IDisposable).Dispose()
    Assert.True((cache ()).Disposed)
