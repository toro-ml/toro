namespace Toro

/// All errors that Toro operations can produce.
type ToroError =
    | TensorNotFound of path: string
    | UnsupportedDType of string
    | UnsupportedDevice of string
    | Msg of string
    /// A .NET exception caught at the TorchSharp boundary.
    | TorchSharpError of exn

    override this.ToString() =
        match this with
        | TensorNotFound path -> $"TensorNotFound: {path}"
        | UnsupportedDType dt -> $"UnsupportedDType: {dt}"
        | UnsupportedDevice dev -> $"UnsupportedDevice: {dev}"
        | Msg msg -> msg
        | TorchSharpError ex -> $"TorchSharpError: {ex.Message}"

module ToroError =
    /// Run f, catching any exception as a TorchSharpError.
    let inline wrap ([<InlineIfLambda>] f: unit -> 'a) : Result<'a, ToroError> =
        try
            Ok(f ())
        with ex ->
            Error(TorchSharpError ex)

    let msg s : Result<'a, ToroError> = Error(Msg s)

type ResultBuilder() =
    member _.Return x = Ok x
    member _.ReturnFrom x = x

    member _.Bind(m, f) =
        match m with
        | Ok x -> f x
        | Error e -> Error e

    member _.Zero() = Ok()

    member _.Combine(m: Result<unit, 'e>, f: unit -> Result<'b, 'e>) =
        match m with
        | Ok() -> f ()
        | Error e -> Error e

    member _.Delay f = f
    member _.Run f = f ()

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

    member this.While(guard, body) =
        if not (guard ()) then
            this.Zero()
        else
            this.Bind(body (), (fun () -> this.While(guard, body)))

    member this.For(sequence: seq<'a>, body) =
        this.Using(sequence.GetEnumerator(), fun enum -> this.While(enum.MoveNext, this.Delay(fun () -> body enum.Current)))

[<AutoOpen>]
module ResultCE =
    let result = ResultBuilder()

[<AutoOpen>]
module Pipe =
    /// Kleisli composition for Result: feeds the Ok output of f into g.
    let inline (>=>)
        ([<InlineIfLambda>] f: 'a -> Result<'b, 'e>)
        ([<InlineIfLambda>] g: 'b -> Result<'c, 'e>)
        : 'a -> Result<'c, 'e> =
        fun a -> f a |> Result.bind g

module Option =
    let inline traverseResult ([<InlineIfLambda>] f: 'a -> Result<'b, 'e>) (opt: 'a option) : Result<'b option, 'e> =
        match opt with
        | Some x -> f x |> Result.map Some
        | None -> Ok None

module List =
    /// Map each element through a Result-returning function, collecting Ok values.
    /// Short-circuits on the first Error. Preserves order in O(n).
    let traverseResult (f: 'a -> Result<'b, 'e>) (xs: 'a list) : Result<'b list, 'e> =
        List.foldBack
            (fun x acc ->
                match f x with
                | Error e -> Error e
                | Ok v ->
                    match acc with
                    | Error e -> Error e
                    | Ok rest -> Ok(v :: rest))
            xs
            (Ok [])
