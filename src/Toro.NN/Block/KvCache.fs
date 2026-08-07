namespace Toro.NN

open Toro

type KvCache(dim: int) =
    let mutable kData: Tensor option = None
    let mutable vData: Tensor option = None
    let mutable currentSeqLen: int = 0

    member _.Dim = dim

    member _.CurrentSeqLen = currentSeqLen

    member _.reset() =
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

            kData <- Some newK
            vData <- Some newV
            currentSeqLen <- currentSeqLen + seqLen
            return newK, newV
        }

    member _.currentData() : (Tensor * Tensor) option =
        match kData, vData with
        | Some k, Some v -> Some(k, v)
        | _ -> None

module KvCache =
    let create (dim: int) : KvCache = KvCache(dim)
