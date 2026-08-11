namespace Toro

open TorchSharp

/// Compute device for tensor storage.
type Device =
    /// CPU device.
    | Cpu
    /// CUDA device with the given index.
    | Cuda of int

module Device =
    /// Convert to a TorchSharp device.
    let toTorch (device: Device) : torch.Device =
        match device with
        | Cpu -> torch.CPU
        | Cuda n -> new torch.Device(DeviceType.CUDA, n)

    /// Try to convert a TorchSharp device. Return None for unsupported types.
    let tryOfTorch (device: torch.Device) : Device option =
        match device.``type`` with
        | DeviceType.CPU -> Some Cpu
        | DeviceType.CUDA -> Some(Cuda device.index)
        | _ -> None

    /// Convert a TorchSharp device. Raise on unsupported types.
    let ofTorch (device: torch.Device) : Device =
        match tryOfTorch device with
        | Some d -> d
        | None -> raise (System.NotSupportedException $"Unsupported device: {device}")
