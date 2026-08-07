namespace Toro

type ToroError =
    | ShapeMismatch of msg: string * expected: int list * got: int list
    | DTypeMismatch of msg: string
    | DeviceError of msg: string
    | TensorNotFound of path: string
    | UnsupportedDType of string
    | UnsupportedDevice of string
    | Msg of string
    | Wrapped of exn

    override this.ToString() =
        match this with
        | ShapeMismatch(msg, expected, got) ->
            $"ShapeMismatch: {msg} \
              (expected {expected}, got {got})"
        | DTypeMismatch msg -> $"DTypeMismatch: {msg}"
        | DeviceError msg -> $"DeviceError: {msg}"
        | TensorNotFound path -> $"TensorNotFound: {path}"
        | UnsupportedDType dt -> $"UnsupportedDType: {dt}"
        | UnsupportedDevice dev -> $"UnsupportedDevice: {dev}"
        | Msg msg -> msg
        | Wrapped ex -> $"Error: {ex.Message}"

module ToroError =
    let inline wrap ([<InlineIfLambda>] f: unit -> 'a) : Result<'a, ToroError> =
        try
            Ok(f ())
        with ex ->
            Error(Wrapped ex)

    let msg s : Result<'a, ToroError> = Error(Msg s)

type ResultBuilder() =
    member _.Return(x) = Ok x
    member _.ReturnFrom(x) = x

    member _.Bind(m, f) =
        match m with
        | Ok x -> f x
        | Error e -> Error e

    member _.Zero() = Ok()

    member _.Combine(m: Result<unit, 'e>, f: unit -> Result<'b, 'e>) =
        match m with
        | Ok() -> f ()
        | Error e -> Error e

    member _.Delay(f) = f
    member _.Run(f) = f ()

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

module Option =
    let inline traverseResult ([<InlineIfLambda>] f: 'a -> Result<'b, 'e>) (opt: 'a option) : Result<'b option, 'e> =
        match opt with
        | Some x -> f x |> Result.map Some
        | None -> Ok None
