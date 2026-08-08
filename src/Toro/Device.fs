namespace Toro

open TorchSharp

/// Compute device for tensor storage.
type Device =
    | Cpu
    | Cuda of int

module Device =
    let toTorch (device: Device) : torch.Device =
        match device with
        | Cpu -> torch.CPU
        | Cuda n -> new torch.Device(DeviceType.CUDA, n)

    let tryOfTorch (device: torch.Device) : Result<Device, ToroError> =
        match device.``type`` with
        | DeviceType.CPU -> Ok Cpu
        | DeviceType.CUDA -> Ok(Cuda device.index)
        | dt -> Error(UnsupportedDevice(string dt))

    let ofTorch (device: torch.Device) : Device =
        match tryOfTorch device with
        | Ok d -> d
        | Error e -> raise (System.NotSupportedException(string e))
