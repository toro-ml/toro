namespace Toro.NN

open TorchSharp
open Toro

type KvCache(dim: int) =
    let mutable kData: Tensor option = None
    let mutable vData: Tensor option = None
    let mutable currentSeqLen: int64 = 0L

    member _.Dim = dim

    member _.CurrentSeqLen = currentSeqLen

    member _.reset() =
        kData |> Option.iter _.Dispose()
        vData |> Option.iter _.Dispose()
        kData <- None
        vData <- None
        currentSeqLen <- 0L

    member _.append(k: Tensor, v: Tensor) : Tensor * Tensor =
        let seqLen = k.shape[dim]

        let newK =
            match kData with
            | None -> k
            | Some prev -> torch.cat ([| prev; k |], int64 dim)

        let newV =
            match vData with
            | None -> v
            | Some prev -> torch.cat ([| prev; v |], int64 dim)

        let oldK = kData
        let oldV = vData
        kData <- Some newK
        vData <- Some newV
        currentSeqLen <- currentSeqLen + seqLen

        oldK
        |> Option.iter (fun t ->
            if not (obj.ReferenceEquals(t, newK)) then
                t.Dispose())

        oldV
        |> Option.iter (fun t ->
            if not (obj.ReferenceEquals(t, newV)) then
                t.Dispose())

        Tensor.keep newK, Tensor.keep newV

    member _.currentData() : (Tensor * Tensor) option =
        match kData, vData with
        | Some k, Some v -> Some(k, v)
        | _ -> None

    interface System.IDisposable with
        member this.Dispose() = this.reset ()

module KvCache =
    let create (dim: int) : KvCache = new KvCache(dim)
