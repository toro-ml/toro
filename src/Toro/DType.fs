namespace Toro

open TorchSharp

/// Element data type of a tensor.
type DType =
    | F16
    | BF16
    | F32
    | F64
    | I32
    | I64
    | U8
    | Bool

module DType =
    let toTorch (dtype: DType) : torch.ScalarType =
        match dtype with
        | F16 -> torch.ScalarType.Float16
        | BF16 -> torch.ScalarType.BFloat16
        | F32 -> torch.ScalarType.Float32
        | F64 -> torch.ScalarType.Float64
        | I32 -> torch.ScalarType.Int32
        | I64 -> torch.ScalarType.Int64
        | U8 -> torch.ScalarType.Byte
        | Bool -> torch.ScalarType.Bool

    let tryOfTorch (dtype: torch.ScalarType) : Result<DType, ToroError> =
        match dtype with
        | torch.ScalarType.Float16 -> Ok F16
        | torch.ScalarType.BFloat16 -> Ok BF16
        | torch.ScalarType.Float32 -> Ok F32
        | torch.ScalarType.Float64 -> Ok F64
        | torch.ScalarType.Int32 -> Ok I32
        | torch.ScalarType.Int64 -> Ok I64
        | torch.ScalarType.Byte -> Ok U8
        | torch.ScalarType.Bool -> Ok Bool
        | dt -> Error(UnsupportedDType(string dt))

    let ofTorch (dtype: torch.ScalarType) : DType =
        match tryOfTorch dtype with
        | Ok d -> d
        | Error e -> raise (System.NotSupportedException(string e))
