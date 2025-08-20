{
  lib,
  builder,
  linkerScript,
  t1main,
  buddy-mlir,
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
        sed "s/STEP_PLACEHOLDER/$vLen/g" ./saxpy.mlir > $caseName.mlir
        cp $caseName.mlir $out/

        echo "[nix] Lowering MLIR"
        buddy-opt $caseName.mlir \
          --lower-affine \
          --convert-vector-to-scf \
          --convert-scf-to-cf \
          --convert-cf-to-llvm=index-bitwidth=32 \
          --convert-vector-to-llvm=index-bitwidth=32 \
          --convert-index-to-llvm=index-bitwidth=32 \
          --convert-math-to-llvm \
          --convert-arith-to-llvm=index-bitwidth=32 \
          --cse \
          --llvm-request-c-wrappers \
          --convert-func-to-llvm=index-bitwidth=32 \
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
          -mattr=+m,+f,+zvl4096b,+zve32f \
          --filetype=obj \
          -O3 \
          -o "$caseName.o"

        echo "Building final binary"
        $CXX -O3 -D SAXPY_LMUL=$vLen -nostdlib -I${../include} -c saxpy.cc -o host.o
        $CC -O3 -T${linkerScript} \
          host.o $caseName.o ${t1main} \
          -o $pname.elf
      '';
    };

  build_platform = import ./platform.nix { inherit lib buddy-mlir builder; };
in
{
  saxpy_8 = build "saxpy_8" 8;
  saxpy_16 = build "saxpy_16" 16;
  saxpy_32 = build "saxpy_32" 32;
  saxpy_64 = build "saxpy_64" 64;
  saxpy_128 = build "saxpy_128" 128;
  saxpy_256 = build "saxpy_256" 256;
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
