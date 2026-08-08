namespace Toro

open TorchSharp

/// Element data type of a tensor.
type DType =
    /// IEEE 754 half-precision (16-bit) float.
    | F16
    /// Brain floating-point (16-bit) float.
    | BF16
    /// Single-precision (32-bit) float.
    | F32
    /// Double-precision (64-bit) float.
    | F64
    /// Signed 32-bit integer.
    | I32
    /// Signed 64-bit integer.
    | I64
    /// Unsigned 8-bit integer.
    | U8
    /// Boolean.
    | Bool

module DType =
    /// Convert to a TorchSharp scalar type.
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

    /// Try to convert a TorchSharp scalar type. Return Error for unsupported types.
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

    /// Convert a TorchSharp scalar type. Raise on unsupported types.
    let ofTorch (dtype: torch.ScalarType) : DType =
        match tryOfTorch dtype with
        | Ok d -> d
        | Error e -> raise (System.NotSupportedException(string e))
