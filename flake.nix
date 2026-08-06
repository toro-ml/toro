{
  description = "Toro";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs =
    inputs@{
      self,
      nixpkgs,
      flake-parts,
    }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
        "x86_64-darwin"
      ];
      perSystem =
        { system, ... }:
        let
          pkgs = import nixpkgs { inherit system; };
        in
        {
          devShells.default = pkgs.mkShell {
            buildInputs = [
              pkgs.dotnet-sdk_10
            ];

            DOTNET_CLI_TELEMETRY_OPTOUT = "1";
            DOTNET_NOLOGO = "1";

            shellHook =
              let
                isDarwin = pkgs.stdenv.hostPlatform.isDarwin;
                arch =
                  if isDarwin then
                    (if pkgs.stdenv.hostPlatform.isAarch64 then "osx-arm64" else "osx-x64")
                  else
                    (if pkgs.stdenv.hostPlatform.isAarch64 then "linux-arm64" else "linux-x64");
                ldVar = if isDarwin then "DYLD_LIBRARY_PATH" else "LD_LIBRARY_PATH";
              in
              ''
                NUGET_PACKAGES="''${NUGET_PACKAGES:-$HOME/.nuget/packages}"
                _torch_paths=""
                for d in "$NUGET_PACKAGES"/libtorch-cpu-${arch}/*/runtimes/${arch}/native \
                         "$NUGET_PACKAGES"/torchsharp/*/runtimes/${arch}/native; do
                  if [ -d "$d" ]; then
                    _torch_paths="$_torch_paths:$d"
                  fi
                done
                if [ -n "$_torch_paths" ]; then
                  export ${ldVar}="''${_torch_paths#:}''${${ldVar}:+:$${ldVar}}"
                fi
                unset _torch_paths
              '';
          };
        };
    };
}
