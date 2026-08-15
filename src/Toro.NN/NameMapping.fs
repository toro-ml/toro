namespace Toro.NN

open System
open System.Collections.Generic

type private PatternSegment =
    | PatternLiteral of string
    | Capture of string

type private TemplateSegment =
    | TemplateLiteral of string
    | Reference of string

/// The result of resolving an external tensor name.
type NameResolution =
    | Keep
    | Rename of targetName: string
    | Ignore

/// A validated declaration for translating or ignoring external tensor names.
type NameRule =
    private
    | ExactRename of source: string * target: string * description: string
    | PatternRewrite of
        source: PatternSegment array *
        target: TemplateSegment array *
        structuralKey: string *
        description: string
    | SuffixIgnore of suffix: string array * description: string

/// A validated set of unambiguous name mapping declarations.
type NameMapping = private NameMapping of NameRule list

module private NameSyntax =

    let segments (argumentName: string) (value: string) =
        if String.IsNullOrWhiteSpace value then
            invalidArg argumentName "A tensor name cannot be empty."

        let result = value.Split('.', StringSplitOptions.None)

        if result |> Array.exists String.IsNullOrEmpty then
            invalidArg argumentName $"Tensor name '{value}' contains an empty path segment."

        result

    let isIdentifier (value: string) =
        not (String.IsNullOrEmpty value)
        && (Char.IsLetter value[0] || value[0] = '_')
        && (value
            |> Seq.skip 1
            |> Seq.forall (fun character -> Char.IsLetterOrDigit character || character = '_'))

    let tryPlaceholder (argumentName: string) (segment: string) =
        let opens = segment.StartsWith('{')
        let closes = segment.EndsWith('}')

        if opens && closes then
            let name = segment[1 .. segment.Length - 2]

            if not (isIdentifier name) then
                invalidArg argumentName $"Placeholder '{{{name}}}' is not a valid identifier."

            Some name
        elif segment.Contains('{') || segment.Contains('}') then
            invalidArg argumentName $"Placeholder in segment '{segment}' must occupy the complete path segment."
        else
            None

    let pattern (argumentName: string) (value: string) =
        let captures = HashSet<string>(StringComparer.Ordinal)

        let parsed =
            segments argumentName value
            |> Array.map (fun segment ->
                match tryPlaceholder argumentName segment with
                | Some name when captures.Add name -> Capture name
                | Some name -> invalidArg argumentName $"Source pattern '{value}' declares capture '{name}' more than once."
                | None -> PatternLiteral segment)

        if captures.Count = 0 then
            invalidArg argumentName $"Source pattern '{value}' has no captures; use NameRule.rename for an exact name."

        parsed, captures

    let template (argumentName: string) (value: string) (captures: HashSet<string>) =
        segments argumentName value
        |> Array.map (fun segment ->
            match tryPlaceholder argumentName segment with
            | Some name when captures.Contains name -> Reference name
            | Some name -> invalidArg argumentName $"Target template '{value}' refers to unknown capture '{name}'."
            | None -> TemplateLiteral segment)

    let structuralKey (segments: PatternSegment array) =
        segments
        |> Array.map (function
            | PatternLiteral value -> $"L:{value}"
            | Capture _ -> "C")
        |> String.concat "."

/// Constructors for validated name mapping declarations.
module NameRule =

    /// Rename one exact external tensor name.
    let rename (source: string) (target: string) : NameRule =
        NameSyntax.segments (nameof source) source |> ignore
        NameSyntax.segments (nameof target) target |> ignore
        ExactRename(source, target, $"rename '{source}' -> '{target}'")

    /// Rewrite a dot-separated name. Placeholders such as {layer} must occupy complete segments.
    let rewrite (sourcePattern: string) (targetTemplate: string) : NameRule =
        let source, captures = NameSyntax.pattern (nameof sourcePattern) sourcePattern
        let target = NameSyntax.template (nameof targetTemplate) targetTemplate captures
        let key = NameSyntax.structuralKey source

        PatternRewrite(source, target, key, $"rewrite '{sourcePattern}' -> '{targetTemplate}'")

    /// Ignore names ending in the specified dot-separated path segments.
    let ignoreSuffix (suffix: string) : NameRule =
        let parsed = NameSyntax.segments (nameof suffix) suffix
        SuffixIgnore(parsed, $"ignore suffix '{suffix}'")

/// Creation and deterministic resolution of name mappings.
module NameMapping =

    let private duplicateKey rule =
        match rule with
        | ExactRename(source, _, _) -> $"exact:{source}"
        | PatternRewrite(_, _, structuralKey, _) -> $"pattern:{structuralKey}"
        | SuffixIgnore(suffix, _) -> "suffix:" + String.concat "." suffix

    let private description rule =
        match rule with
        | ExactRename(_, _, value)
        | PatternRewrite(_, _, _, value)
        | SuffixIgnore(_, value) -> value

    /// Validate and create a mapping. Duplicate declarations are rejected.
    let create (rules: NameRule list) : NameMapping =
        let duplicates =
            rules
            |> List.groupBy duplicateKey
            |> List.choose (fun (_, matching) ->
                match matching with
                | []
                | [ _ ] -> None
                | _ -> Some(matching |> List.map description))

        match duplicates with
        | duplicate :: _ -> invalidArg (nameof rules) $"Duplicate name mapping rules: %A{duplicate}."
        | [] -> NameMapping rules

    /// A mapping that preserves every source name.
    let identity = NameMapping []

    let private tryPattern sourceSegments pattern template =
        if Array.length sourceSegments <> Array.length pattern then
            None
        else
            (Some Map.empty, Array.zip pattern sourceSegments)
            ||> Array.fold (fun captures (segment, value) ->
                captures
                |> Option.bind (fun captures ->
                    match segment with
                    | PatternLiteral expected when value <> expected -> None
                    | PatternLiteral _ -> Some captures
                    | Capture name -> Some(Map.add name value captures)))
            |> Option.map (fun captures ->
                template
                |> Array.map (function
                    | TemplateLiteral value -> value
                    | Reference name -> Map.find name captures)
                |> String.concat "."
                |> Rename)

    let private tryResolve sourceName sourceSegments rule =
        match rule with
        | ExactRename(source, target, _) when String.Equals(source, sourceName, StringComparison.Ordinal) -> Some(Rename target)
        | ExactRename _ -> None
        | PatternRewrite(pattern, template, _, _) -> tryPattern sourceSegments pattern template
        | SuffixIgnore(suffix, _) when
            sourceSegments.Length >= suffix.Length
            && Array.forall2 (=) suffix sourceSegments[sourceSegments.Length - suffix.Length ..]
            ->
            Some Ignore
        | SuffixIgnore _ -> None

    /// Resolve one source name. Unmatched names are kept; multiple matches are rejected.
    let resolve (NameMapping rules) (sourceName: string) : NameResolution =
        let sourceSegments = NameSyntax.segments (nameof sourceName) sourceName

        let matches =
            rules
            |> List.choose (fun rule ->
                tryResolve sourceName sourceSegments rule
                |> Option.map (fun resolution -> rule, resolution))

        match matches with
        | [] -> Keep
        | [ _, resolution ] -> resolution
        | _ ->
            let declarations = matches |> List.map (fst >> description)
            invalidOp $"Name '{sourceName}' matches multiple name mapping rules: %A{declarations}."
