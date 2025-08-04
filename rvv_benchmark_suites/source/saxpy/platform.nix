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
    caseName = "saxpy_${platform}";
    src = ./.;

    nativeBuildInputs = [
      buddy-mlir
      buddy-mlir.llvm
    ];

    buildPhase = ''
      mkdir -p $out

      vs=8
      echo "[$caseName] Using VS=$vs"
      sed "s/STEP_PLACEHOLDER/$vs/g" ./saxpy.mlir > $caseName.mlir
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
  "saxpy_${platform}" = drv;
}
