add_cuda_executable(mmm_bench bench.cu mmm.cu)
target_link_libraries(mmm_bench nvbench::main)

add_cuda_executable(mmm_example example.cu mmm.cu)

install(TARGETS mmm_bench mmm_example)
