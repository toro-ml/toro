namespace Toro.NN

open Toro

type IVarBackend =
    abstract get:
        shape: int list * name: string * init: Init * dtype: DType * device: Device ->
            Result<Tensor, ToroError>

    abstract containsTensor: name: string -> bool

type VarBuilder = {
    Prefix: string
    DType: DType
    Device: Device
    Backend: IVarBackend
}

module VarBuilder =
    let private fullPath (vb: VarBuilder) (name: string) =
        if vb.Prefix.Length = 0 then
            name
        else
            vb.Prefix + "." + name

    let pp (prefix: string) (vb: VarBuilder) : VarBuilder = {
        vb with
            Prefix =
                if vb.Prefix.Length = 0 then
                    prefix
                else
                    vb.Prefix + "." + prefix
    }

    let get
        (shape: int list)
        (name: string)
        (vb: VarBuilder)
        : Result<Tensor, ToroError> =
        let p = fullPath vb name

        vb.Backend.get (shape, p, Init.Const 0.0, vb.DType, vb.Device)

    let getWithHints
        (shape: int list)
        (name: string)
        (init: Init)
        (vb: VarBuilder)
        : Result<Tensor, ToroError> =
        let p = fullPath vb name

        vb.Backend.get (shape, p, init, vb.DType, vb.Device)

    let containsTensor (name: string) (vb: VarBuilder) =
        let p = fullPath vb name
        vb.Backend.containsTensor p

    // --- Backend implementations ---

    type TensorMapBackend(tensors: Map<string, Tensor>) =
        interface IVarBackend with
            member _.get(shape, name, _init, dtype, device) =
                match Map.tryFind name tensors with
                | Some t ->
                    let tShape = t.Shape

                    if tShape <> shape then
                        Error(
                            ShapeMismatch(
                                $"shape mismatch \
                                  for {name}",
                                shape,
                                tShape
                            )
                        )
                    else
                        result {
                            let! t = t.toDType dtype

                            return! t.toDevice device
                        }
                | None -> Error(TensorNotFound name)

            member _.containsTensor name = Map.containsKey name tensors

    type ZerosBackend() =
        interface IVarBackend with
            member _.get(shape, _name, _init, dtype, device) =
                Tensor.zeros (shape, dtype, device)

            member _.containsTensor _ = true

    type InitBackend() =
        interface IVarBackend with
            member _.get(shape, _name, init, dtype, device) =
                Init.toTensor shape dtype device init

            member _.containsTensor _ = true

    let fromTensors
        (tensors: Map<string, Tensor>)
        (dtype: DType)
        (device: Device)
        : VarBuilder =
        {
            Prefix = ""
            DType = dtype
            Device = device
            Backend = TensorMapBackend(tensors)
        }

    let fromZeros (dtype: DType) (device: Device) : VarBuilder = {
        Prefix = ""
        DType = dtype
        Device = device
        Backend = ZerosBackend()
    }

    let fromInit (dtype: DType) (device: Device) : VarBuilder = {
        Prefix = ""
        DType = dtype
        Device = device
        Backend = InitBackend()
    }
