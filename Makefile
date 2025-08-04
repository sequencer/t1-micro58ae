NIX := nix

EMULATOR_BIN_DIR := emulators
RTL_ARCHIVE_DIR := rtls
RVV_CASE_DIR := rvv_benchmark_suites
SVE_CASE_DIR := sve_benchmark_suites

EMU_TYPE := t1rocketemu

BENCHMARK_DESIGNS := benchmark_dlen128_vlen512_fp \
			benchmark_dlen128_vlen1024_fp \
			benchmark_dlen256_vlen1024_fp \
			benchmark_dlen512_vlen1024_fp \
			benchmark_dlen512_vlen2048_fp \
			benchmark_dlen1024_vlen1024_fp \
			benchmark_dlen1024_vlen4096_fp \
			physical_design_case_0 \
			physical_design_case_1 \
			physical_design_case_2 \
			physical_design_case_3 \
			physical_design_case_4 \
			physical_design_case_5 \
			physical_design_case_6 \
			physical_design_case_7 \
			physical_design_case_8 \
			physical_design_case_9 \
			physical_design_case_10 \
			physical_design_case_11 \
			physical_design_case_12 \
			physical_design_case_13 \
			physical_design_case_14 \
			physical_design_case_15 \
			physical_design_case_16 \
			physical_design_case_17 \
			physical_design_case_18 \
			physical_design_case_19 \
			physical_design_case_20 \
			physical_design_case_21 \
			physical_design_case_22 \
			physical_design_case_23 \
			physical_design_case_24 \
			physical_design_case_25 \
			physical_design_case_26 \
			physical_design_case_27 \
			physical_design_case_28

EMULATOR_DESIGNS := benchmark_dlen128_vlen512_fp \
			benchmark_dlen128_vlen1024_fp \
			benchmark_dlen256_vlen1024_fp \
			benchmark_dlen512_vlen1024_fp \
			benchmark_dlen512_vlen2048_fp \
			benchmark_dlen1024_vlen1024_fp \
			benchmark_dlen1024_vlen4096_fp

CASES := memset \
		ascii_to_utf32 \
		byteswap \
		linear_normalization \
		matmul \
		saxpy_8 saxpy_16 \
		sgemm_8 sgemm_64 sgemm_128 \
		quant_8 quant_16 \
		pack_256 pack_1024 \
		ntt_512 \
		ntt_1024 \
		ntt_2048 \
		ntt_4096 \
		mmm_1024 \
		mmm_2048 \
		mmm_4096 \
		mmm_8192

ALL_CASES := $(foreach design,$(EMULATOR_DESIGNS),$(foreach case,$(CASES),$(design).$(case).elf))

.PHONY: rtl
rtl: $(addprefix $(RTL_ARCHIVE_DIR)/,$(BENCHMARK_DESIGNS))

.PHONY: cases
cases: $(addprefix $(RVV_CASE_DIR)/bin/,$(ALL_CASES))

.PHONY: emulator
emulator: $(addsuffix /emulator,$(addprefix $(EMULATOR_BIN_DIR)/,$(EMULATOR_DESIGNS)))

.PHONY: disable_chaining_emulator
disable_chaining_emulator: $(EMULATOR_BIN_DIR)/disable_chaining/emulator

.PHONY: disable_memory_interleaving_emulator
disable_memory_interleaving_emulator: $(EMULATOR_BIN_DIR)/disable_memory_interleaving/emulator

$(RTL_ARCHIVE_DIR)/%:
	$(NIX) build ".#t1.$*.t1.rtl" --out-link result
	mkdir $@
	cp -vLfr ./result/* $@/
	chown -R ${USER}:${USER} $@
	chmod -R u+w $@
	rm -rf $@/verification

$(EMULATOR_BIN_DIR)/disable_chaining/$(EMU_TYPE)/emulator:
	$(NIX) build ".#t1_no_chaining.benchmark_dlen256_vlen4096_fp.$(EMU_TYPE).vcs-emu" --impure --out-link result
	$(NIX) build ".#t1_no_chaining.benchmark_dlen256_vlen4096_fp.$(EMU_TYPE).vcs-dpi-lib" --impure --out-link dpi-lib-result
	mkdir -p $(@D)
	cp -vT ./result/bin/* $@
	cp -vr ./dpi-lib-result/lib $(@D)/
	rm result
	rm dpi-lib-result

$(EMULATOR_BIN_DIR)/disable_memory_interleaving/$(EMU_TYPE)/emulator:
	$(NIX) build ".#t1_no_interleaving.benchmark_dlen256_vlen4096_fp.$(EMU_TYPE).vcs-emu" --impure --out-link result
	$(NIX) build ".#t1_no_interleaving.benchmark_dlen256_vlen4096_fp.$(EMU_TYPE).vcs-dpi-lib" --impure --out-link dpi-lib-result
	mkdir -p $(@D)
	cp -vT ./result/bin/* $@
	cp -vr ./dpi-lib-result/lib $(@D)/
	rm result
	rm dpi-lib-result

$(EMULATOR_BIN_DIR)/%/$(EMU_TYPE)/emulator:
	$(NIX) build ".#t1.$*.$(EMU_TYPE).vcs-emu" --impure --out-link result
	$(NIX) build ".#t1.$*.$(EMU_TYPE).vcs-dpi-lib" --impure --out-link dpi-lib-result
	mkdir -p $(@D)
	cp -vT ./result/bin/* $@
	cp -vr ./dpi-lib-result/lib $(@D)/
	rm result
	rm dpi-lib-result

$(RVV_CASE_DIR)/bin/%.elf:
	mkdir -p $(@D)
	cfg=$(shell echo $* | cut -d'.' -f1); \
	case=$(shell echo $* | cut -d'.' -f2); \
		$(NIX) build ".#$$cfg.t1rocketemu.bench.$$case" -L --out-link result
	cp ./result/bin/*.elf $@

$(RVV_CASE_DIR)/lib_rv64/%.o:
	mkdir -p $(@D)
	nix build '.#benchmark_dlen256_vlen1024_fp.t1rocketemu.bench.$*_riscv64'
	cp ./result/lib/*.o $@

$(RVV_CASE_DIR)/lib_rv64/%.elf: $(RVV_CASE_DIR)/lib_rv64/%.o
	mkdir -p $(@D)
	clang++ -march=rv64gcv -O3 \
		-I ./rvv_benchmark_suites/source/include \
		./rvv_benchmark_suites/source/$*/platform.cc $< \
		-o $@

$(SVE_CASE_DIR)/lib/%.o:
	mkdir -p $(@D)
	nix build '.#benchmark_dlen256_vlen1024_fp.t1rocketemu.bench.$*_aarch64'
	cp ./result/lib/*.o $@

$(SVE_CASE_DIR)/bin/%.elf: $(SVE_CASE_DIR)/lib/%.o
	mkdir -p $(@D)
	clang++ -march=armv8.6-a+sve -O3 \
		-I ./rvv_benchmark_suites/source/include \
		./rvv_benchmark_suites/source/$*/platform.cc $< \
		-o $@
