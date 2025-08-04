final: prev: {
  myCudaPackages = final.cudaPackages_12_8;
  cudaStdenv = final.gcc14Stdenv;

  nvbench = final.callPackage ./nvbench.nix {
    cudaPackages = final.myCudaPackages;
    stdenv = final.cudaStdenv;
  };
}
