namespace Toro.Extensions.AI

open System
open System.Collections.Generic
open System.Threading
open System.Threading.Tasks

module internal AsyncEnumerable =

    type private Enumerator<'T>(enumerator: IEnumerator<'T>, cancellation: CancellationTokenSource) =
        let mutable disposed = false

        let dispose () =
            if not disposed then
                disposed <- true

                try
                    enumerator.Dispose()
                finally
                    cancellation.Dispose()

        let moveNext () =
            try
                cancellation.Token.ThrowIfCancellationRequested()
                let hasNext = enumerator.MoveNext()

                if not hasNext then
                    dispose ()

                hasNext
            with _ ->
                dispose ()
                reraise ()

        interface IAsyncEnumerator<'T> with
            member _.Current = enumerator.Current

            member _.MoveNextAsync() =
                if disposed then
                    ValueTask<bool>(false)
                else
                    ValueTask<bool>(Task.Run(Func<bool>(moveNext)))

            member _.DisposeAsync() =
                dispose ()
                ValueTask()

    type private Enumerable<'T>(requestCancellationToken, source: CancellationToken -> seq<'T>) =
        interface IAsyncEnumerable<'T> with
            member _.GetAsyncEnumerator(enumerationCancellationToken) =
                let cancellation =
                    CancellationTokenSource.CreateLinkedTokenSource(requestCancellationToken, enumerationCancellationToken)

                try
                    let enumerator = (source cancellation.Token).GetEnumerator()
                    new Enumerator<_>(enumerator, cancellation) :> IAsyncEnumerator<'T>
                with _ ->
                    cancellation.Dispose()
                    reraise ()

    let ofBackgroundSeq requestCancellationToken source =
        new Enumerable<_>(requestCancellationToken, source) :> IAsyncEnumerable<_>
