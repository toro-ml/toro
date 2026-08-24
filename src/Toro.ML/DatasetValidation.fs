namespace Toro.ML

open TorchSharp
open Toro

module internal DatasetValidation =

    let features (paramName: string) (value: Tensor) =
        if value.dim () <> 2 then
            invalidArg paramName $"Features must be 2-d, but have {value.dim ()} dimensions."

        if value.dtype <> torch.float32 then
            invalidArg paramName $"Features must have dtype float32, but have {value.dtype}."

        if value.shape[0] <= 0L then
            invalidArg paramName "Features must contain at least one row."

        if value.shape[1] <= 0L then
            invalidArg paramName "Features must contain at least one column."

    let float32Labels (paramName: string) (rowCount: int64) (value: Tensor) =
        if value.dim () <> 1 then
            invalidArg paramName $"Labels must be 1-d, but have {value.dim ()} dimensions."

        if value.dtype <> torch.float32 then
            invalidArg paramName $"Labels must have dtype float32, but have {value.dtype}."

        if value.shape[0] <> rowCount then
            invalidArg paramName $"Labels have {value.shape[0]} rows, expected {rowCount}."
