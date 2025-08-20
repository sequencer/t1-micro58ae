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
    caseName: vlen:
    builder {
      inherit caseName;
      src = ./.;

      nativeBuildInputs = [
        buddy-mlir
        buddy-mlir.llvm
      ];

      buildPhase = ''
        mkdir -p $out

        vLen=${toString vlen}
        echo "[$caseName] Using VLEN=$vLen"
        sed "s/STEP_PLACEHOLDER/$vLen/g" ./sgemm.mlir > $caseName.mlir
        cp $caseName.mlir $out/

        echo "[nix] Lowering MLIR"
        buddy-opt $caseName.mlir \
          --lower-affine \
          --llvm-request-c-wrappers \
          --convert-vector-to-llvm=index-bitwidth=32 \
          --convert-scf-to-cf \
          --convert-cf-to-llvm=index-bitwidth=32 \
          --convert-math-to-llvm \
          --convert-func-to-llvm=index-bitwidth=32 \
          --convert-index-to-llvm=index-bitwidth=32 \
          --convert-arith-to-llvm=index-bitwidth=32 \
          --cse \
          --expand-strided-metadata \
          --finalize-memref-to-llvm=index-bitwidth=32 \
          --reconcile-unrealized-casts \
          -o $caseName-opted.mlir

        echo "[nix] Translate MLIR to LLVM IR"
        buddy-translate --buddy-to-llvmir "$caseName-opted.mlir" -o "$caseName.ll"

        echo "[nix] Compiling LLVM IR to OBJECT"
        llc "$caseName.ll" \
          -mtriple=riscv32 \
          -target-abi=ilp32f \
          -mattr=+m,+f,+zvl${toString rtlDesignMetadata.vlen}b,+zve32f \
          --filetype=obj \
          -O3 \
          -o "$caseName.o"

        echo "Building final binary"
        $CXX -O3 -D SGEMM_LMUL=$vLen -nostdlib -I${../include} -c sgemm.cc -o host.o
        $CC -O3 -T${linkerScript} \
          host.o $caseName.o ${t1main} \
          -o $pname.elf
      '';

      passthru.rv64 = import ./rv64.nix {
        inherit
          buddy-mlir
          caseName
          builder
          ;
      };
    };
  build_platform = import ./platform.nix { inherit lib buddy-mlir builder; };
in
{
  sgemm_8 = build "sgemm_8" 8;
  sgemm_16 = build "sgemm_16" 16;
  sgemm_32 = build "sgemm_32" 32;
  sgemm_64 = build "sgemm_64" 64;
  sgemm_128 = build "sgemm_128" 128;
  sgemm_256 = build "sgemm_256" 256;
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
