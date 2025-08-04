{
  lib,
  buddy-mlir,
  builder,
}:
{
  platform,
  llcArgs ? [ ],
}:

let
  drv = builder {
    caseName = "pack_${platform}";
    src = ./.;

    nativeBuildInputs = [
      buddy-mlir
      buddy-mlir.llvm
    ];

    buildPhase = ''
      mkdir -p $out

      vs=256
      vs4=1024
      echo "[$caseName] Using VS=$vs"
      sed "s/STEP_PLACEHOLDER/$vs/g" ./pack.mlir \
        | sed "s/STEP_4_PLACEHOLDER/$vs4/g" \
        > $caseName.mlir
      cp $caseName.mlir $out/

      echo "[nix] Lowering MLIR"
      buddy-opt $caseName.mlir \
        --lower-affine \
        --llvm-request-c-wrappers \
        --convert-vector-to-llvm \
        --convert-scf-to-cf \
        --convert-cf-to-llvm \
        --convert-math-to-llvm \
        --convert-func-to-llvm \
        --convert-index-to-llvm \
        --convert-arith-to-llvm \
        --cse \
        --expand-strided-metadata \
        --finalize-memref-to-llvm \
        --reconcile-unrealized-casts \
        -o $caseName-opted.mlir

      echo "[nix] Translate MLIR to LLVM IR"
      buddy-translate --buddy-to-llvmir "$caseName-opted.mlir" -o "$caseName.ll"

      echo "[nix] Compiling LLVM IR to OBJECT"
      llc "$caseName.ll" --filetype=obj -O3 ${lib.escapeShellArgs llcArgs} -o "$pname.o"
    '';
  };
in
{
  "pack_${platform}" = drv;
}
