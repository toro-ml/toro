# Agent guidelines

## Dev Environment

- Run all commands through `nix develop -c` (e.g. `nix develop -c dotnet build`). The flake shell sets `DYLD_LIBRARY_PATH`/`LD_LIBRARY_PATH` for TorchSharp native libraries.
- Build: `nix develop -c dotnet build`
- Test: `nix develop -c dotnet test`
- Run an example: `nix develop -c dotnet run --project examples/<Name>`

## Format

- Run `dotnet fantomas` to format code (`src/`, `tests/`, `examples/`).
- Lefthook formats staged `*.fs` / `*.fsi` / `*.fsx` on commit (`nix develop` installs the hook).
- Follow F# 10 coding style.
- Follow ASD-STE100 for technical English.

## Docs

- Regenerate API references: run `pnpm generate:api` in `docs/`.
- When adding a package: update `filePages` in `scripts/generate-api-mdx.mts` and `docs/app/ui/nav/nav-data.ts`.
