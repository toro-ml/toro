namespace rec Toro

open TorchSharp

/// Type alias exposing TorchSharp tensor under the shorter Toro name.
type Tensor = torch.Tensor

/// Index specifier for <c>Tensor.at</c>. I=single, S=slice, A=all, T=tensor, E=ellipsis, N=newaxis.
type TIdx =
    | I of int
    | S of start: int * stop: int
    | Sf of start: int
    | St of stop: int
    | A
    | T of torch.Tensor
    | E
    | N

[<AutoOpen>]
module TensorOps =
    /// Convert a float to a TorchSharp Scalar for use with tensor operators.
    let scalar (v: float) : Scalar = Scalar.op_Implicit v

    /// Elementwise $a = b$.
    let (.=.) (a: torch.Tensor) (b: torch.Tensor) = a.eq b

    /// Elementwise $a \neq b$.
    let (.<>.) (a: torch.Tensor) (b: torch.Tensor) = a.ne b

    /// Elementwise $a > b$.
    let (.>.) (a: torch.Tensor) (b: torch.Tensor) = a.gt b

    /// Elementwise $a < b$.
    let (.<.) (a: torch.Tensor) (b: torch.Tensor) = a.lt b

    /// Elementwise $a \geq b$.
    let (.>=.) (a: torch.Tensor) (b: torch.Tensor) = a.ge b

    /// Elementwise $a \leq b$.
    let (.<=.) (a: torch.Tensor) (b: torch.Tensor) = a.le b

[<AutoOpen>]
module TensorExtensions =
    type torch.Tensor with

        /// Index by a list of TIdx specifiers.
        member this.at(indices: TIdx list) : torch.Tensor =
            let toTorchIndex =
                function
                | TIdx.I i -> torch.TensorIndex.Single(int64 i)
                | TIdx.S(s, e) -> torch.TensorIndex.Slice(System.Nullable(int64 s), System.Nullable(int64 e))
                | TIdx.Sf s -> torch.TensorIndex.Slice(System.Nullable(int64 s), System.Nullable())
                | TIdx.St e -> torch.TensorIndex.Slice(System.Nullable(), System.Nullable(int64 e))
                | TIdx.A -> torch.TensorIndex.Slice()
                | TIdx.T t -> torch.TensorIndex.Tensor(t)
                | TIdx.E -> torch.TensorIndex.Ellipsis
                | TIdx.N -> torch.TensorIndex.None

            let tIndices = indices |> List.toArray |> Array.map toTorchIndex
            this.index tIndices

        /// Return the accumulated gradient, or zeros if none.
        member this.grad() =
            if isNull this.grad then
                torch.zeros_like this
            else
                this.grad

        /// Zero the accumulated gradient.
        member this.zeroGrad() =
            if not (isNull this.grad) then
                this.grad.zero_ () |> ignore

        /// Copy data from src without gradient tracking.
        member this.copyInPlace(src: torch.Tensor) =
            use _scope = torch.no_grad ()
            this.copy_ src |> ignore

module Tensor =
    /// Move a tensor out of the current dispose scope so it
    /// survives when that scope exits. Inside <c>scoped { }</c>,
    /// return values are auto-kept, so <c>keep</c> is only needed
    /// for side-effect retention (e.g. caching a tensor in a mutable field).
    let keep (t: torch.Tensor) : torch.Tensor =
        match torch.CurrentDisposeScope with
        | null -> ()
        | scope ->
            if scope.Contains(t) then
                scope.MoveToOuter(t) |> ignore

        t

module Toro =
    /// Run f with gradient tracking disabled.
    let noGrad (f: unit -> 'a) : 'a =
        use _scope = torch.no_grad ()
        f ()

    /// Run f in inference mode (faster than noGrad; disables view tracking).
    let inferenceMode (f: unit -> 'a) : 'a =
        use _scope = torch.inference_mode ()
        f ()

[<AutoOpen>]
module internal DisposeScopeHelper =
    let private flags =
        System.Reflection.BindingFlags.Public
        ||| System.Reflection.BindingFlags.NonPublic

    let rec keepTensors (scope: DisposeScope) (v: obj) =
        if isNull v then
            ()
        else
            match v with
            | :? torch.Tensor as t ->
                if scope.Contains(t) then
                    scope.MoveToOuter(t) |> ignore
            | :? System.Collections.IEnumerable as xs ->
                for item in xs do
                    keepTensors scope item
            | _ ->
                let ty = v.GetType()

                if Microsoft.FSharp.Reflection.FSharpType.IsTuple ty then
                    for field in Microsoft.FSharp.Reflection.FSharpValue.GetTupleFields v do
                        keepTensors scope field
                elif Microsoft.FSharp.Reflection.FSharpType.IsRecord(ty, flags) then
                    for field in Microsoft.FSharp.Reflection.FSharpValue.GetRecordFields(v, flags) do
                        keepTensors scope field
                elif Microsoft.FSharp.Reflection.FSharpType.IsUnion(ty, flags) then
                    let _, fields = Microsoft.FSharp.Reflection.FSharpValue.GetUnionFields(v, ty, flags)

                    for field in fields do
                        keepTensors scope field
                elif ty.IsValueType && ty.IsGenericType then
                    for prop in
                        ty.GetProperties(
                            System.Reflection.BindingFlags.Public
                            ||| System.Reflection.BindingFlags.Instance
                        ) do
                        let pv = prop.GetValue(v)

                        if not (isNull pv) then
                            keepTensors scope pv

/// Computation expression that wraps the body in a
/// <c>torch.NewDisposeScope()</c>. Intermediate tensors are disposed
/// automatically when the block completes. Tensors in the return value
/// (including inside tuples, records, and unions) are kept alive past the scope.
type ScopedBuilder() =
    member _.Return(x) = x
    member _.ReturnFrom(x) = x
    member _.Zero() = ()

    member _.Combine(_: unit, f: unit -> 'b) : 'b = f ()

    member _.Delay(f: unit -> 'a) = f

    member _.Run(f: unit -> 'a) : 'a =
        use scope = torch.NewDisposeScope()
        let v = f ()
        keepTensors scope (box v)
        v

    member _.TryWith(body, handler) =
        try
            body ()
        with ex ->
            handler ex

    member _.TryFinally(body, finalizer) =
        try
            body ()
        finally
            finalizer ()

    member _.Using(resource: #System.IDisposable, body) =
        try
            body resource
        finally
            if not (isNull (box resource)) then
                resource.Dispose()

    member _.While(guard, body) =
        while guard () do
            body () |> ignore

    member this.For(sequence: seq<'a>, body) =
        this.Using(sequence.GetEnumerator(), fun enum -> this.While(enum.MoveNext, this.Delay(fun () -> body enum.Current)))

/// Computation expression that disposes every tensor created in its scope unless
/// the tensor is moved to the outer scope explicitly with <c>Tensor.keep</c>.
/// Unlike <c>scoped</c>, the return value is not traversed automatically.
type ExplicitScopedBuilder() =
    member _.Return(x) = x
    member _.ReturnFrom(x) = x
    member _.Zero() = ()

    member _.Combine(_: unit, f: unit -> 'b) : 'b = f ()

    member _.Delay(f: unit -> 'a) = f

    member _.Run(f: unit -> 'a) : 'a =
        use _scope = torch.NewDisposeScope()
        f ()

    member _.TryWith(body, handler) =
        try
            body ()
        with ex ->
            handler ex

    member _.TryFinally(body, finalizer) =
        try
            body ()
        finally
            finalizer ()

    member _.Using(resource: #System.IDisposable, body) =
        try
            body resource
        finally
            if not (isNull (box resource)) then
                resource.Dispose()

    member _.While(guard, body) =
        while guard () do
            body () |> ignore

    member this.For(sequence: seq<'a>, body) =
        this.Using(sequence.GetEnumerator(), fun enum -> this.While(enum.MoveNext, this.Delay(fun () -> body enum.Current)))

[<AutoOpen>]
module ScopedCE =
    /// Computation expression that wraps the body in a
    /// <c>torch.NewDisposeScope()</c>. Intermediate tensors are disposed
    /// automatically when the block completes.
    let scoped = ScopedBuilder()

    /// Computation expression that disposes intermediate tensors without inspecting
    /// the return value. Call <c>Tensor.keep</c> for each tensor that must survive.
    let scopedExplicit = ExplicitScopedBuilder()
