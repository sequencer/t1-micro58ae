{
  lib,
  newScope,
  rv32-stdenv,
  rtlDesignMetadata,
  python3,
  linkerScript,
  t1main,
  buddy-mlir,
  bc,
}:
lib.makeScope newScope (
  scope:
  let
    searchDir = "source";

    builder = scope.callPackage ./builder.nix {
      stdenv = rv32-stdenv;
      inherit rtlDesignMetadata;
    };
  in
  lib.recurseIntoAttrs (
    (lib.pipe (builtins.readDir ./${searchDir}) [
      (lib.filterAttrs (
        name: type:
        type == "directory"
        && !(lib.hasPrefix "_" name)
        && !(lib.elem name [
          "lib"
          "include"
        ])
      ))
      (lib.mapAttrs (subDirName: _: (lib.path.append ./${searchDir} subDirName)))
      (lib.attrValues)
      (lib.map (
        srcPath:
        import srcPath {
          inherit
            rtlDesignMetadata
            buddy-mlir
            bc
            builder
            linkerScript
            t1main
            python3
            lib
            ;
        }
      ))
      (lib.flatten)
      (lib.foldr (acc: elem: acc // elem) { })
    ])
  ) # end of recurseIntoAttrs
) # end of scope
