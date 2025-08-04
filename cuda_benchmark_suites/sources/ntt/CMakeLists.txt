add_cuda_executable(ntt_bench bench.cu ntt.cu)
target_link_libraries(ntt_bench nvbench::main)

add_cuda_executable(ntt_example example.cu ntt.cu)

install(TARGETS ntt_bench ntt_example)
