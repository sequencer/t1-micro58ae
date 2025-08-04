{
  builder,
  linkerScript,
  t1main,
  ...
}:
{
  matmul = builder rec {
    caseName = "matmul";
    src = ./.;

    buildPhase = ''
      runHook preBuild

      $CC -T${linkerScript} \
        ${caseName}.c \
        ${t1main} \
        -o $pname.elf

      runHook postBuild
    '';
  };
}
