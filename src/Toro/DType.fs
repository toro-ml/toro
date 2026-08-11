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

    /// Try to convert a TorchSharp scalar type. Return None for unsupported types.
    let tryOfTorch (dtype: torch.ScalarType) : DType option =
        match dtype with
        | torch.ScalarType.Float16 -> Some F16
        | torch.ScalarType.BFloat16 -> Some BF16
        | torch.ScalarType.Float32 -> Some F32
        | torch.ScalarType.Float64 -> Some F64
        | torch.ScalarType.Int32 -> Some I32
        | torch.ScalarType.Int64 -> Some I64
        | torch.ScalarType.Byte -> Some U8
        | torch.ScalarType.Bool -> Some Bool
        | _ -> None

    /// Convert a TorchSharp scalar type. Raise on unsupported types.
    let ofTorch (dtype: torch.ScalarType) : DType =
        match tryOfTorch dtype with
        | Some d -> d
        | None -> raise (System.NotSupportedException $"Unsupported dtype: {dtype}")
