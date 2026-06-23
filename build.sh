#!/bin/sh
# Copyright (c) 2026 Peter Fors
# SPDX-License-Identifier: MIT
set -e
export PATH="$HOME/.cargo/bin:$PATH"

# Competitor/oracle: the genuine image-rs/image-png (ext/image-png, unmodified,
# .git pins the commit) built with portable_simd via the staticlib shim. cargo
# no-ops when nothing changed, so this stays cheap on rebuilds.
( cd ext/shim && cargo +nightly build --release )

# Our decoder: portable x86-64-v3 (AVX2 + BMI2), aggressive alignment, PIE/unwind
# /stack-protector overhead stripped. -O2 (O3 is worse); generic-tuned gcc beats
# clang and znver4 tuning on this scalar hot loop.
DECFLAGS="-std=c99 -O2 -march=x86-64-v3 -mbmi -mbmi2 -mpclmul \
	-falign-functions=32 -falign-loops=32 -fno-plt \
	-fwrapv -fno-stack-protector -fno-pie -fcf-protection=none \
	-ffunction-sections -fdata-sections -fno-unwind-tables -fno-asynchronous-unwind-tables \
	-U_FORTIFY_SOURCE -Wall -Wextra"

cc $DECFLAGS -c ffpng.c -o ffpng.o
cc -O2 -march=native -Wall -Wextra -c bench.c -o bench.o
cc -no-pie -Wl,--gc-sections -o bench bench.o ffpng.o \
	ext/shim/target/release/libimagepng_shim.a \
	-lpng -lz -lm -lpthread -ldl
