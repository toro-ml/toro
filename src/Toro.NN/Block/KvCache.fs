namespace Toro.NN

open Toro

type KvCache(dim: int) =
    let mutable kData: Tensor option = None
    let mutable vData: Tensor option = None
    let mutable currentSeqLen: int = 0

    member _.Dim = dim

    member _.CurrentSeqLen = currentSeqLen

    member _.reset() =
        kData |> Option.iter _.Dispose()
        vData |> Option.iter _.Dispose()
        kData <- None
        vData <- None
        currentSeqLen <- 0

    member _.append(k: Tensor, v: Tensor) : Result<Tensor * Tensor, ToroError> =
        result {
            let seqLen = k.Shape[dim]

            let! newK =
                match kData with
                | None -> Ok k
                | Some prev -> Tensor.cat ([ prev; k ], dim)

            let! newV =
                match vData with
                | None -> Ok v
                | Some prev -> Tensor.cat ([ prev; v ], dim)

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

            return Tensor.keep newK, Tensor.keep newV
        }

    member _.currentData() : (Tensor * Tensor) option =
        match kData, vData with
        | Some k, Some v -> Some(k, v)
        | _ -> None

    interface System.IDisposable with
        member this.Dispose() = this.reset ()

module KvCache =
    let create (dim: int) : KvCache = new KvCache(dim)
