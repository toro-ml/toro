module TestHelper

open Toro
open TorchSharp

let scalarF32 (t: Tensor) = (t.sum ()).ToSingle()

let scalarF64 (t: Tensor) = (t.sum ()).ToDouble()
