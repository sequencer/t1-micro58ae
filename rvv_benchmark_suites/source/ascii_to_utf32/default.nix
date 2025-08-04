{
  builder,
  linkerScript,
  t1main,
  ...
}:
{
  ascii_to_utf32 = builder rec {
    caseName = "ascii_to_utf32";
    src = ./.;

    buildPhase = ''
      runHook preBuild

      cp -r ${../_rvv-bench} rvv-bench
      pushd rvv-bench >/dev/null
      $CC -E -DINC=$(realpath ../$caseName.S) template.S -E -o ../functions.S
      popd >/dev/null

      $CC -Irvv-bench ${caseName}.c -T${linkerScript} ${t1main} functions.S -o $pname.elf

      runHook postBuild
    '';
  };
}
