namespace Toro.Models

open TorchSharp
open Toro

type internal PreparedCausalInput = {
    SequenceLength: int64
    CacheStart: int64
    PositionIds: Tensor
    AttentionMask: Tensor option
    IsCausal: bool
}

module internal CausalInput =

    let private attentionMask (input: CausalLmInput<'Cache>) batchSize sequenceLength start totalLength =
        let paddingMask =
            input.AttentionMask
            |> Option.map (fun mask ->
                if mask.shape <> [| batchSize; totalLength |] then
                    invalidArg (nameof input.AttentionMask) $"Attention mask shape must be [{batchSize}, {totalLength}]."

                mask.to_type(torch.ScalarType.Bool).unsqueeze(1L).unsqueeze (1L))

        let needsExplicitCausalMask = start > 0L && sequenceLength > 1L

        match paddingMask, needsExplicitCausalMask with
        | None, false -> None, start = 0L
        | Some padding, false when sequenceLength = 1L -> Some padding, false
        | padding, _ ->
            let keyPositions =
                torch.arange (totalLength, dtype = torch.int64, device = input.InputIds.device)

            let queryPositions =
                torch.arange (start, start + sequenceLength, dtype = torch.int64, device = input.InputIds.device)

            let causal =
                (keyPositions.unsqueeze (0L)
                 .<=. queryPositions.unsqueeze (1L))
                    .unsqueeze(0L)
                    .unsqueeze (0L)

            match padding with
            | Some value -> Some(causal.logical_and value), false
            | None -> Some causal, false

    let prepare
        (modelName: string)
        (maxPositions: int64)
        (cacheLength: 'Cache -> int64)
        (validateCache: 'Cache -> int64 -> int64 -> unit)
        (input: CausalLmInput<'Cache>)
        : PreparedCausalInput =
        if input.InputIds.dtype <> torch.int64 then
            invalidArg (nameof input.InputIds) $"{modelName} input IDs must use int64 dtype."

        if input.InputIds.shape.Length <> 2 then
            invalidArg (nameof input.InputIds) $"{modelName} input IDs must have shape [batch, sequence]."

        let batchSize = input.InputIds.shape[0]
        let sequenceLength = input.InputIds.shape[1]

        if batchSize <= 0L || sequenceLength <= 0L then
            invalidArg (nameof input.InputIds) $"{modelName} input dimensions must be positive."

        let cacheStart =
            input.Cache
            |> Option.map cacheLength
            |> Option.defaultValue 0L

        input.Cache
        |> Option.iter (fun cache -> validateCache cache batchSize sequenceLength)

        if cacheStart + sequenceLength > maxPositions then
            invalidArg (nameof input.InputIds) $"{modelName} accepts at most {maxPositions} positions."

        let positionIds =
            match input.PositionIds with
            | None ->
                torch.arange (cacheStart, cacheStart + sequenceLength, dtype = torch.int64, device = input.InputIds.device)
            | Some positions ->
                if positions.dtype <> torch.int64 then
                    invalidArg (nameof input.PositionIds) $"{modelName} position IDs must use int64 dtype."

                if
                    positions.shape <> [| sequenceLength |]
                    && positions.shape <> [| batchSize; sequenceLength |]
                then
                    invalidArg
                        (nameof input.PositionIds)
                        $"{modelName} position IDs must have shape [sequence] or [batch, sequence]."

                positions

        let totalLength = cacheStart + sequenceLength

        let mask, isCausal =
            attentionMask input batchSize sequenceLength cacheStart totalLength

        {
            SequenceLength = sequenceLength
            CacheStart = cacheStart
            PositionIds = positionIds
            AttentionMask = mask
            IsCausal = isCausal
        }
