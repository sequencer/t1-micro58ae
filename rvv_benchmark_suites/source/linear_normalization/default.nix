{
  builder,
  linkerScript,
  t1main,
  ...
}:
{
  linear_normalization = builder rec {
    caseName = "linear_normalization";
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
