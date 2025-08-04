{
  description = "cuda for crypto";

  inputs = {
    nixpkgs.url = "nixpkgs";
    flake-utils.url = "flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }@inputs:
    flake-utils.lib.eachDefaultSystem
      (system:
        let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [
              (import ./nix/overlay.nix)
            ];
            config.allowUnfree = true;
          };
          cudaPackages = pkgs.myCudaPackages;
        in
        rec {
          legacyPackages = pkgs;

          # only gcc stdenv is supported, clangStdenv produces linker error
          defaultPackage = pkgs.cudaStdenv.mkDerivation {
            name = "cuda_crypto";

            nativeBuildInputs = with pkgs; [ cmake ninja ];
            buildInputs = with pkgs; [
              cudaPackages.cudatoolkit
              cudaPackages.cuda_cudart
              cudaPackages.cuda_nvml_dev
              nvbench
              thrust

              autoAddDriverRunpath
            ];

            src = with pkgs.lib.fileset; toSource {
              root = ./.;
              fileset = fileFilter
                (
                  file: ! (pkgs.lib.elem file.name [ "flake.nix" "flake.lock" ])
                ) ./.;
            };
          };

          devShell = defaultPackage.overrideAttrs (oldAttrs: {
            # https://github.com/NixOS/nixpkgs/issues/214945
            nativeBuildInputs = (oldAttrs.nativeBuildInputs or [ ]) ++ (with pkgs; [
              clang-tools
            ]);

            shellHook = ''
              export NIX_CFLAGS_COMPILE="$NIX_CFLAGS_COMPILE -fdiagnostics-color=always"
              export NIX_LDFLAGS="$NIX_LDFLAGS -rpath /run/opengl-driver/lib"
              export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/run/opengl-driver/lib:${pkgs.xorg.libXtst}/lib:${pkgs.systemd}/lib"
            '';
          });

          devShells.fhs = defaultPackage.overrideAttrs (oldAttrs: {
            # https://github.com/NixOS/nixpkgs/issues/214945
            nativeBuildInputs = (oldAttrs.nativeBuildInputs or [ ]) ++ (with pkgs; [
              clang-tools
            ]);

            shellHook = ''
              export NIX_CFLAGS_COMPILE="$NIX_CFLAGS_COMPILE -fdiagnostics-color=always"
              export NIX_LDFLAGS="$NIX_LDFLAGS -L /usr/lib/x86_64-linux-gnu -rpath /usr/local/cuda-12.2/targets/x86_64-linux/lib"
            '';
          });

        }
      )
    // {
      inherit inputs; # for easier introspection via nix repl
    };
}

