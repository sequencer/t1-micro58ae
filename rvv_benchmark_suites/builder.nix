{
  lib,
  stdenv,
  jq,
  rtlDesignMetadata,
}:

{ caseName, ... }@overrides:

stdenv.mkDerivation (
  lib.recursiveUpdate rec {
    name = "t1bench.${caseName}";
    pname = name;

    CC = "${stdenv.targetPlatform.config}-cc";
    CXX = "${stdenv.targetPlatform.config}-c++";
    AR = "${stdenv.targetPlatform.config}-ar";
    OBJDUMP = "${stdenv.targetPlatform.config}-objdump";

    NIX_CFLAGS_COMPILE =
      let
        march = lib.pipe rtlDesignMetadata.march [
          (lib.splitString "_")
          (map (ext: if ext == "zvbb" then "zvbb1" else ext))
          (lib.concatStringsSep "_")
        ];
      in
      [
        "-mabi=ilp32f"
        "-march=${march}"
        "-mno-relax"
        "-static"
        "-mcmodel=medany"
        "-fvisibility=hidden"
        "-fno-PIC"
        "-g"
        "-O3"
      ];

    installPhase = ''
      runHook preInstall

      if [[ -f "$pname.elf" ]]; then
        mkdir -p $out/bin
        cp -v $pname.elf $out/bin
        $OBJDUMP -d $out/bin/$pname.elf > $out/$pname.s
      elif [[ -f "$pname.o" ]]; then
        mkdir -p $out/lib
        cp -v $pname.o $out/lib
        $OBJDUMP -d $out/lib/$pname.o > $out/$pname.s
      else
        echo "No file found"
        exit 1
      fi

      runHook postInstall
    '';

    dontFixup = true;

    passthru = {
      inherit rtlDesignMetadata;
    };
  } overrides
)
