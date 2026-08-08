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

    /// Try to convert a TorchSharp device. Return Error for unsupported types.
    let tryOfTorch (device: torch.Device) : Result<Device, ToroError> =
        match device.``type`` with
        | DeviceType.CPU -> Ok Cpu
        | DeviceType.CUDA -> Ok(Cuda device.index)
        | dt -> Error(UnsupportedDevice(string dt))

    /// Convert a TorchSharp device. Raise on unsupported types.
    let ofTorch (device: torch.Device) : Device =
        match tryOfTorch device with
        | Ok d -> d
        | Error e -> raise (System.NotSupportedException(string e))
