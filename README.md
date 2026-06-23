# ffpng

A from-scratch C PNG decoder for x86-64, built to beat `image-rs/image-png`, the
Rust decoder behind the 2026 "fastest PNG decoder in the world" post. Over the
whole QOI benchmark corpus it decodes about 1.17x faster than image-png by
geometric mean, on a Ryzen 7950X with image-png built in its strongest
configuration, and it does so with only AVX2 against image-png's AVX-512 (see
below). It is not faster on every image: on dense, high-entropy photographs it
still loses (see Numbers).

Decode only, no encoder. Output is image-png's native `EXPAND | STRIP_16`: 8-bit
samples, channels native to the color type, for every PNG color type and bit
depth (palette, grayscale, RGB, RGBA, 1/2/4/8/16-bit, interlaced). Output is
verified byte-for-byte against image-png on all 2848 corpus images.

## How it works

The decoder is one translation unit, `ffpng.c`, libc and `<immintrin.h>` only,
no third-party libraries.

**Inflate** is most of the work and is where the speed is won or lost. A 64-bit
refill bit reader feeds a two-level Huffman scheme: a 9-bit direct table for
short codes with a canonical maxcode fallback for the rare long ones. For images
whose raw (filtered) size clears 128 KB it also builds a 12-bit *combined* table
whose every entry resolves in a single load to one or two literals, a match
length base, or end-of-block. The literal path runs a four-deep speculative
cascade: the next three table loads are issued from the still-unconsumed bit
buffer so their L1 latencies pipeline, and up to three literals are emitted
before one consume-and-refill. This structure is modeled on `fdeflate`,
image-png's inflate backend.

The combined table is built by incremental doubling: each symbol is placed once
at its codeword, adjacent literal pairs are fused, and the table is doubled with
one copy per added bit, instead of a strided per-symbol scatter. The match copy
is a branchless 16-byte SIMD over-copy (a 16-byte tail slack on every buffer
keeps the over-write in bounds) rather than a `memcpy` call.

**CRC-32** folds the 16-byte-aligned bulk with `PCLMULQDQ` (the reflected
algorithm), and finishes the sub-16-byte tail from a table.

**Unfilter** is SSE/AVX2. Up is a flat 32-byte vertical add. Sub, Average, and
Paeth are per-pixel, since the left-neighbour dependency is serial; Sub and
Average run in 8-bit lanes (the Average floor is `avg_epu8(a,b) - ((a^b)&1)`),
Paeth in 16-bit lanes.

For large non-interlaced images the inflate, unfilter, and expand passes are
**fused and streamed**: a row is unfiltered in place and expanded to pixels as
soon as it falls a full 32 KB window behind the inflate write frontier (so it
can no longer be back-referenced), while it is still hot in cache, instead of a
second cold pass over the raw buffer.

The cascade's trade-off shows up in the per-category numbers. On match-heavy
data (textures, screenshots, icons) ffpng's inflate is faster than fdeflate's;
on dense-literal photographs it is a few percent slower, because the literal
cascade is then the critical path and ffpng extracts slightly less
instruction-level parallelism per load.

## Build

```sh
git submodule update --init     # fetch the image-png competitor
./build.sh
```

Needs:
- `gcc` (tested 16.1) targeting `x86-64-v3` (AVX2 + BMI2) with PCLMUL
- Rust `nightly` + `cargo` (image-png's `unstable` feature requires it)
- `libpng` and `zlib` (the benchmark also times libpng as a reference)

`build.sh` builds the image-png shim staticlib with cargo (a no-op when nothing
changed), compiles `ffpng.o` and `bench.o`, and links `./bench`. The competitor is
the genuine, unmodified `image-rs/image-png` repository, vendored as a git
submodule at `ext/image-png` pinned to the exact commit benchmarked (`4ab5484`,
v0.18.1). It is built with `--features=unstable` and `target-cpu=native`, its
fastest portable-SIMD configuration, so the comparison is against image-png at
its best, not a hobbled build.

## AVX2, not AVX-512

The interesting part is *how* ffpng wins. The image-png build benchmarked above,
its fastest, leans hard on AVX-512 when compiled `target-cpu=native` on Zen 4:
its inflate, unfilter, and checksum code emit hundreds of AVX-512 instructions,
including the upper `zmm16`-`zmm31` registers that exist only in AVX-512. ffpng
emits none. It is pure AVX2 (`-march=x86-64-v3`). Both are easy to check:

```sh
objdump -d ext/shim/target/release/libimagepng_shim.a | grep -c '%zmm'   # ~950
objdump -d ffpng.o                                     | grep -c '%zmm'   # 0
```

So ffpng beats an AVX-512 decoder using only AVX2, and that is the practically
useful result. One `x86-64-v3` binary runs at full speed on every AVX2 CPU
(Intel since 2013, AMD since 2017); reaching image-png's benchmarked numbers
instead needs an AVX-512 build, which will not even start on those machines.
AVX-512 was tried for ffpng and measured slower: Zen 4 double-pumps it, and the
inflate hot loop is scalar bit-twiddling that does not widen, so giving it up
costs nothing here and buys portability.

## Benchmark

```sh
taskset -c 1 chrt -f 99 ./bench [--decoder NAME] [--csv FILE] images
python3 analyze.py FILE          # per-category ratios from a CSV
```

Pin to one core at real-time priority, and disable CPU boost, for stable
numbers. `bench` registers four decoders (image-png, this one, stb_image,
libpng), times each on every image, and verifies correctness: the first decoder
(image-png) is the oracle; every other decoder's output is converted to
canonical RGBA8 outside the timing loop and `memcmp`'d against it.

Timing is decode only, the geometric mean of per-image megapixels/second over
many iterations per image; per-decode allocation is inside the timed region
because it is a real cost. The per-image *ratio* against image-png is the
portable, machine-independent comparison, since both decoders are timed
microseconds apart on the same image; absolute MP/s vary with CPU and boost or
thermal state.

## Numbers

Ryzen 7950X, single core at real-time priority, gcc 16.1, image-png v0.18.1
`--features=unstable`, 2848 images. Sorted worst-first. `ffpng` and `image-png`
are geomean megapixels/second; `ratio` is `ffpng / image-png`.

| category        | ffpng | image-png | ratio |    N |
|-----------------|------:|----------:|------:|-----:|
| photo_tecnick   | 213.9 |     220.9 | 0.968 |  100 |
| photo_wikipedia | 166.7 |     156.4 | 1.066 |   49 |
| textures_photo  | 159.5 |     149.1 | 1.070 |   20 |
| icon_64         | 245.6 |     228.0 | 1.077 |  213 |
| photo_kodak     | 160.9 |     148.6 | 1.083 |   24 |
| textures_pk02   | 188.1 |     172.0 | 1.094 |  235 |
| textures_plants | 272.9 |     244.3 | 1.117 |   60 |
| textures_pk01   | 228.9 |     203.3 | 1.126 |  113 |
| pngimg          | 340.7 |     298.9 | 1.140 |  187 |
| textures_pk     | 429.2 |     367.2 | 1.169 | 1002 |
| screenshot_game | 374.4 |     300.9 | 1.244 |  618 |
| icon_512        | 586.0 |     447.4 | 1.310 |  213 |
| screenshot_web  | 597.2 |     400.9 | 1.490 |   14 |
| **overall**     | **344.4** | **295.2** | **1.167** | 2848 |

The honest read: `photo_tecnick` is the one category we lose, by ~3%. It is the
densest-literal set in the corpus (1200x1200 RGB, only ~1.8x compressible), so
its decode is almost entirely the inflate literal cascade, and there fdeflate
edges us. The two highest-weight categories, `textures_pk` (35% of the corpus)
and `screenshot_game` (22%), are both wins and carry the geomean. Everything
except `photo_tecnick` is faster than image-png.

## Layout

```
ffpng.c  ffpng.h    the decoder (one TU; libc + immintrin only)
bench.c             benchmark driver: registers decoders, times, checks output
build.sh            builds the shim, then ffpng.o + bench
analyze.py          per-category ratios from a bench CSV
ext/image-png/      genuine image-rs/image-png, git submodule pinned at 4ab5484
ext/shim/           Rust staticlib wrapping image-png for the C benchmark
ext/stb_image.h     stb_image, third-party (another decoder the bench times)
images/             the corpus (not vendored; see below)
```

## The corpus

The benchmark images are the QOI benchmark suite, the same set image-png's post
measured: photographs, textures, screenshots, icons, and assorted PNGs across 13
categories. They are third-party and are not vendored here. The tarball already
stores its files as `images/<category>/*.png`, so from the repository root it
unpacks straight into place, no rearranging:

```sh
curl -O https://qoiformat.org/benchmark/qoi_benchmark_suite.tar   # ~1.1 GB
tar xf qoi_benchmark_suite.tar                                    # creates images/
taskset -c 1 chrt -f 99 ./bench --csv /tmp/r.csv images
python3 analyze.py /tmp/r.csv
```

`./bench images` recurses one level into the category directories, which is
exactly the layout the tarball produces. Confirm the suite's licensing before
redistributing it; that is why it is kept out of this repository.
