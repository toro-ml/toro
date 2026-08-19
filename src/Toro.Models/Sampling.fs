namespace Toro.Models

open System
open System.Threading

/// Token-selection strategy used during generation.
type GenerationSampling =
    /// Select the highest-logit token.
    | Greedy
    /// Sample from logits divided by a positive temperature.
    | Temperature of temperature: float

/// Options for one causal language-model generation session.
type GenerationOptions = {
    /// Maximum number of tokens to generate after the prompt.
    MaxNewTokens: int
    /// Token-selection strategy applied to each next-token distribution.
    Sampling: GenerationSampling
    /// Cancellation signal checked before each model invocation.
    CancellationToken: CancellationToken
}

/// Constructors and validation for generation options.
module GenerationOptions =

    /// Create greedy generation options without cancellation.
    let greedy maxNewTokens = {
        MaxNewTokens = maxNewTokens
        Sampling = Greedy
        CancellationToken = CancellationToken.None
    }

    /// Create temperature-sampling options without cancellation.
    let temperature temperature maxNewTokens = {
        MaxNewTokens = maxNewTokens
        Sampling = Temperature temperature
        CancellationToken = CancellationToken.None
    }

    let internal validate options =
        if options.MaxNewTokens < 0 then
            invalidArg (nameof options) "Maximum new-token count must be non-negative."

        match options.Sampling with
        | Greedy -> ()
        | Temperature value when value > 0.0 && Double.IsFinite value -> ()
        | Temperature _ -> invalidArg (nameof options) "Sampling temperature must be finite and positive."
