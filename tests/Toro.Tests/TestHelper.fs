module TestHelper

open Toro

let scalarF32 (t: Tensor) = (t.sumAll ()).toFloat32Scalar ()

let scalarF64 (t: Tensor) = (t.sumAll ()).toFloat64Scalar ()
