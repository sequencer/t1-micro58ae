{
  lib,
  builder,
  linkerScript,
  t1main,
  buddy-mlir,
  rtlDesignMetadata,
  ...
}:
let
  build =
    caseName: vl:
    builder {
      inherit caseName;
      src = ./.;

      nativeBuildInputs = [
        buddy-mlir
        buddy-mlir.llvm
      ];

      buildPhase = ''
        mkdir -p $out

        vLen=${toString vl}
        echo "[nix] Using VLEN=$vLen"
        sed "s/STEP_PLACEHOLDER/$vLen/g" ./quant.mlir > $caseName.mlir
        cp $caseName.mlir $out/

        echo "[nix] Lowering MLIR"
        buddy-opt $caseName.mlir \
          --lower-affine \
          --convert-vector-to-llvm=index-bitwidth=32 \
          --convert-scf-to-cf \
          --convert-cf-to-llvm=index-bitwidth=32 \
          --convert-math-to-llvm \
          --convert-index-to-llvm=index-bitwidth=32 \
          --convert-arith-to-llvm=index-bitwidth=32 \
          --llvm-request-c-wrappers \
          --convert-func-to-llvm=index-bitwidth=32 \
          --cse \
          --expand-strided-metadata \
          --finalize-memref-to-llvm=index-bitwidth=32 \
          --reconcile-unrealized-casts \
          -o $caseName-opted.mlir

        echo "[nix] Translate MLIR to LLVM IR"
        mlir-translate --mlir-to-llvmir "$caseName-opted.mlir" -o "$caseName.ll"

        echo "[nix] Compiling LLVM IR to OBJECT"
        llc "$caseName.ll" \
          -mtriple=riscv32 \
          -target-abi=ilp32f \
          -mattr=+m,+f,+zvl${toString rtlDesignMetadata.vlen}b,+zve32f \
          --filetype=obj \
          -O3 \
          -o "$caseName.o"

        echo "Building final binary"
        $CXX -O3 -D QUANT_LMUL=$vLen -nostdlib -I${../include} -c quant.cc -o host.o
        $CC -O3 -T${linkerScript} \
          host.o $caseName.o ${t1main} \
          -o $pname.elf
      '';

      passthru.rv64 = import ./rv64.nix {
        inherit
          caseName
          buddy-mlir
          builder
          ;
      };
    };
  build_platform = import ./platform.nix { inherit lib buddy-mlir builder; };
in
{
  quant_8 = build "quant_8" 8;
  quant_16 = build "quant_16" 16;
}
// (build_platform {
  platform = "aarch64";
  llcArgs = [
    "-march=aarch64"
    "-mattr=+v8.6a,+sve"
  ];
})
// (build_platform {
  platform = "riscv64";
  llcArgs = [
    "-mtriple=riscv64"
    "-mattr=+v"
  ];
})
