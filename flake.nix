{
  description = "Generic devshell setup";

  inputs = {
    t1.url = "github:chipsalliance/t1/resnet";
    zaozi.url = "github:sequencer/zaozi";
    mif.url = "github:Avimitin/mill-ivy-fetcher";
    nixpkgs.follows = "t1/nixpkgs";
    flake-utils.follows = "t1/flake-utils";
    t1_no_chaining.url = "github:chipsalliance/t1/disable-chaining";
    t1_no_interleaving.url = "github:chipsalliance/t1/single-LSU";
  };

  outputs =
    inputs@{
      self,
      t1,
      t1_no_chaining,
      t1_no_interleaving,
      nixpkgs,
      flake-utils,
      zaozi,
      mif,
    }:
    let
      overlay = (import ./nix/overlay.nix) {
        t1 = t1.outPath;
        t1_no_chaining = t1_no_chaining.outPath;
        t1_no_interleaving = t1_no_interleaving.outPath;
      };
    in
    {
      # System-independent attr
      inherit inputs;
      overlays.default = overlay;
    }
    // flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [
            t1.overlays.default
            zaozi.overlays.default
            mif.overlays.default
            overlay
          ];
        };
      in
      {
        formatter = pkgs.nixpkgs-fmt;
        legacyPackages = pkgs;
      }
    );
}
