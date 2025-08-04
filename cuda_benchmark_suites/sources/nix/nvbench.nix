{ lib
, stdenv
, fetchFromGitHub
, cmake
, cudaPackages
, cpm-cmake
, fmt
, nlohmann_json
}:

let
  rapids-cmake = fetchFromGitHub {
    owner = "rapidsai";
    repo = "rapids-cmake";
    rev = "v25.06.00a";
    hash = "sha256-YC/Y7xWo9Lkb+YZv4ocL5Yvm5FgEzdHAEXZdC3hlXW8=";
  };
in
stdenv.mkDerivation rec {
  pname = "nvbench";
  version = "unstable-2025-04-04";

  src = fetchFromGitHub {
    owner = "NVIDIA";
    repo = "nvbench";
    rev = "1efed5f8e13903a12d9348dab2f7ff0fe6b8ecfd";
    hash = "sha256-BsZcyESZWteCCtxfOyLgvSovXV+PXV5OJqQAk7w2v9o=";
  };

  nativeBuildInputs = [
    cmake
  ];

  buildInputs = [
    cudaPackages.cudatoolkit
    cudaPackages.cuda_cudart
    cudaPackages.cuda_nvml_dev
    cpm-cmake
    fmt
    nlohmann_json
  ];

  cmakeFlags = [
    "-Drapids-cmake-dir=${rapids-cmake}/rapids-cmake"
    "-DCMAKE_MODULE_PATH=${rapids-cmake}/rapids-cmake"
    "-DCPM_DOWNLOAD_LOCATION=${cpm-cmake}/share/cpm/CPM.cmake"
  ];

  patches = [ ./nvbench.patch ];

  meta = {
    description = "CUDA Kernel Benchmarking Library";
    homepage = "https://github.com/NVIDIA/nvbench";
    license = lib.licenses.asl20;
    mainProgram = "nvbench";
    platforms = lib.platforms.all;
  };
}
