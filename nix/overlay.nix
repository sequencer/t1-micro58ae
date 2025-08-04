{
  t1,
  t1_no_chaining,
  t1_no_interleaving,
}:

final: prev:
let
  t1DesignsConfigsDir = "${t1}/designs";

  lib = prev.lib;
  allConfigs = lib.pipe t1DesignsConfigsDir [
    builtins.readDir
    # => { "filename" -> "filetype" }
    (lib.mapAttrs' (
      fileName: fileType:
      assert fileType == "regular" && lib.hasSuffix ".toml" fileName;
      lib.nameValuePair (lib.removeSuffix ".toml" fileName) "${t1DesignsConfigsDir}/${fileName}"
    ))
    # => { "filename without .toml suffix, use as generator className" -> "realpath to config" }
    (lib.mapAttrs (_: filePath: with builtins; fromTOML (readFile filePath)))
    # => { "generator className" -> { "elaborator configName" -> "elaborator cmd opt" } }
    lib.attrsToList
    # => [ { name: "generator className", value: { "configName": { "cmd opt": <string> } } }, ... ]
    (map (
      kv:
      lib.mapAttrs' (
        configName: configData: lib.nameValuePair configName { "${kv.name}" = configData; }
      ) kv.value
    ))
    # => [ { "configName": { "generator className" = { cmdopt: <string>; }; } }, ... ]
    (lib.foldl (accum: item: lib.recursiveUpdate accum item) { })
    # => { "configName A": { "generator A" = { cmdopt }, "generator B" = { cmdopt } }; "configName B" = ...; ... }
  ];

  forEachConfig =
    attrBuilder:
    lib.mapAttrs (configName: allGenerators: attrBuilder configName allGenerators) allConfigs;
in
(forEachConfig (
  configName: allGenerators:
  let
    strippedGeneratorData = lib.mapAttrs' (
      fullClassName: origData:
      lib.nameValuePair (lib.head (
        lib.splitString "." (lib.removePrefix "org.chipsalliance.t1.elaborator." fullClassName)
      )) (origData // { inherit fullClassName; })
    ) allGenerators;

    forEachTop =
      scopeBuilderFn:
      lib.mapAttrs (
        topName: generatorData:
        lib.makeScope final.newScope (
          scope: lib.recurseIntoAttrs (scopeBuilderFn topName generatorData scope)
        )
      ) strippedGeneratorData;
  in
  forEachTop (
    topName: generator: self: rec {
      inherit (final.t1.${configName}.${topName}) rtlDesignMetadata;

      linkerScript = ../rvv_benchmark_suites/source/t1.ld;
      t1main = ../rvv_benchmark_suites/source/t1_main.S;
      bench = self.callPackage ../rvv_benchmark_suites { inherit linkerScript t1main; };
    }
  )
)) # END of forEachConfig

// {
  t1_no_chaining = final.callPackage "${t1_no_chaining}/nix/t1" { };
  t1_no_interleaving = final.callPackage "${t1_no_interleaving}/nix/t1" { };
  # For old T1
  circt-full = final.circt-install;

  buddy-mlir =
    let
      pkgSrc = final.fetchFromGitHub {
        owner = "NixOS";
        repo = "nixpkgs";
        rev = "2a725d40de138714db4872dc7405d86457aa17ad";
        hash = "sha256-WWNNjCSzQCtATpCFEijm81NNG1xqlLMVbIzXAiZysbs=";
      };
      lockedNixpkgs = import pkgSrc { system = final.system; };
    in
    lockedNixpkgs.callPackage ./buddy-mlir.nix { python3 = lockedNixpkgs.python312; };
}
