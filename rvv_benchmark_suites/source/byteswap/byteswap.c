/* https://github.com/camel-cdr/rvv-bench/blob/main/bench/byteswap.c */

#include "bench.h"

#if __riscv_zbb
void
byteswap32_SWAR_rev8(uint32_t *ptr, size_t n)
{
	while (n--) {
		*ptr = __builtin_bswap32(*ptr);
		++ptr;
		BENCH_CLOBBER();
	}
}
#define REV8(f) f(SWAR_rev8)
#else
#define REV8(f)
#endif


/* we don't support these on XTheadVector */
#ifndef __riscv_vector
#define IMPLS_RVV(f)
#else
#define IMPLS_RVV(f) \
	f(rvv_gatherei16_m1) \
	f(rvv_gatherei16_m2) \
	f(rvv_gatherei16_m4) \
	f(rvv_m1_gatherei16s_m2) \
	f(rvv_m1_gatherei16s_m4) \
	f(rvv_m1_gatherei16s_m8)
#endif

#if __riscv_zvbb
#define IMPLS_ZVBB(f) MX(f,rvv_vrev8)
#else
#define IMPLS_ZVBB(f)
#endif


#define IMPLS(f) \
	REV8(f) \
	IMPLS_ZVBB(f) \
	IMPLS_RVV(f)

typedef void Func(uint32_t *ptr, size_t n);

#define DECLARE(f) extern Func byteswap32_##f;
IMPLS(DECLARE)

#define EXTRACT(f) { #f, &byteswap32_##f },
Impl impls[] = { IMPLS(EXTRACT) };

uint32_t *ptr;

void init(void) { ptr = (uint32_t*)mem; }

ux checksum(size_t n) {
	ux sum = 0;
	for (size_t i = 0; i < n; ++i)
		sum = uhash(sum) + ptr[i];
	return sum;
}

BENCH_BEG(base) {
	bench_memrand(ptr, n * sizeof *ptr);
	TIME(f(ptr, n));
} BENCH_END

Bench benches[] = {
	BENCH( impls, MAX_MEM/4, "byteswap32", bench_base )
}; BENCH_MAIN(benches)

