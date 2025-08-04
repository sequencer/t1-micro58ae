{
  builder,
  linkerScript,
  t1main,
  ...
}:
{
  "byteswap" = builder rec {
    caseName = "byteswap";
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
