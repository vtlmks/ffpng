# ffpng

A from-scratch C PNG decoder for x86-64, built to beat `image-rs/image-png`, the
Rust decoder behind the 2026 ["fastest PNG decoder in the world" post](https://blog.image-rs.org/2026/06/18/png-adoption.html). Over the
whole QOI benchmark corpus it decodes about 1.07x faster than image-png by
geometric mean, on a Ryzen 7950X, with both decoders compiled to the same AVX2
instruction set (`x86-64-v3`), which is image-png's fastest build on Zen 4 (see
below). The one set it loses is the densest-entropy photographs, where the
decode is almost entirely serial literal work and image-png's inflate edges it
by a few percent (see Numbers).

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
image-png's inflate backend. Each entry keeps its code length in the low byte,
so advancing the bit buffer to the next speculative load is a single shift with
no extraction: the serial literal-decode chain, which is the whole cost on
dense-literal photographs, stays one shift per symbol.

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

The per-category spread tracks how match-heavy the data is. On match-heavy data
(textures, screenshots, icons) ffpng's inflate is ahead of fdeflate's; on
dense-literal photographs, where the serial literal cascade is the whole
critical path, fdeflate edges it by a few percent.

## Build

```sh
git submodule update --init     # fetch the image-png competitor
./build.sh
```

Needs:
- `gcc` (tested 16.1) targeting `x86-64-v3` (AVX2 + BMI2) with PCLMUL
- Rust `nightly` + `cargo` (image-png's `unstable` feature requires it)
- `libpng` and `zlib` (the benchmark also times libpng as a reference)
- `libdeflate` and `libspng` (two more reference decoders the benchmark times)

Use gcc for the numbers above. ffpng compiles cleanly with clang too, but on
this workload clang runs about 10% slower: it schedules the SIMD unfilter worse
and inlines the match path less aggressively than gcc.

`build.sh` builds the image-png shim staticlib with cargo (a no-op when nothing
changed), compiles `ffpng.o`, `ldpng.o`, and `bench.o`, and links `./bench`. The competitor is
the genuine, unmodified `image-rs/image-png` repository, vendored as a git
submodule at `ext/image-png` pinned to the exact commit benchmarked (`4ab5484`,
v0.18.1). It is built with `--features=unstable` and `target-cpu=x86-64-v3`, the
same AVX2 instruction set as ffpng and image-png's fastest configuration on Zen 4
(`target-cpu=native` emits AVX-512, which measured ~9% slower; see below). So the
comparison is against image-png at its best, on equal instruction set, not a
hobbled build.

## AVX2, not AVX-512

Both decoders are benchmarked as pure AVX2, and that is deliberate: on Zen 4,
AVX-512 is a net loss for this workload. Compiled `target-cpu=native`, image-png
emits hundreds of AVX-512 instructions, including the upper `zmm16`-`zmm31`
registers that exist only in AVX-512; compiled `target-cpu=x86-64-v3` it emits
none, and so does ffpng:

```sh
# the benchmarked x86-64-v3 builds: no AVX-512
objdump -d ext/shim/target/release/libimagepng_shim.a | grep -c '%zmm'   # 0
objdump -d ffpng.o                                     | grep -c '%zmm'   # 0
# rebuild the shim with target-cpu=native and image-png emits ~950 zmm uses
```

The AVX-512 build of image-png runs about 9% slower on this corpus than its
x86-64-v3 build, measured with ffpng held fixed across the two runs as a drift
anchor (the same ffpng binary, decoding to the byte, anchors out clock and
thermal drift between them). Zen 4 double-pumps 512-bit operations, and a PNG
decoder is scalar bit-twiddling in the inflate hot loop plus short-vector
unfiltering, neither of which gains from the extra width. So image-png is
benchmarked at its faster x86-64-v3 build, not the slower native one, and ffpng
is x86-64-v3 for the same reason. The payoff is portability with nothing left on
the table: one binary runs at full speed on every AVX2 CPU (Intel since 2013,
AMD since 2017), and the AVX-512 path it skips would only have been slower here.

## Benchmark

```sh
taskset -c 1 chrt -f 99 ./bench [--decoder NAME] [--csv FILE] images
python3 analyze.py FILE          # per-category ratios from a CSV
```

Pin to one core at real-time priority, and disable CPU boost, for stable
numbers. `bench` registers six decoders, times each on every image, and
verifies correctness: the first decoder (image-png) is the oracle; every other
decoder's output is converted to canonical RGBA8 outside the timing loop and
`memcmp`'d against it.

Without `--decoder`, all six run. `--decoder NAME` restricts the run to one,
where `NAME` is one of:

| `NAME`      | decoder                                       |
|-------------|-----------------------------------------------|
| `image-png` | image-rs/image-png (the oracle)               |
| `ffpng`     | this decoder                                  |
| `ldpng`     | libdeflate's inflate behind this decoder's front end |
| `libspng`   | libspng (miniz inflate)                       |
| `stb_image` | stb_image.h                                   |
| `libpng`    | libpng + zlib                                 |

The other flags: `--csv FILE` writes per-image results, `--filter SUBSTR` keeps
only paths containing `SUBSTR`, `--limit N` caps the corpus to the first `N`
images, and `--min-time-ms` / `--min-iters` / `--max-iters` tune the per-image
timing budget.

Timing is decode only, the geometric mean of per-image megapixels/second over
many iterations per image; per-decode allocation is inside the timed region
because it is a real cost. The per-image *ratio* against image-png is the
portable, machine-independent comparison, since both decoders are timed
microseconds apart on the same image; absolute MP/s vary with CPU and boost or
thermal state.

## Measurement

Small changes are hard to measure because run-to-run noise (clock drift, thermal,
code-layout shifts) is often larger than the effect, so a real 1% gain can look
like noise or even a regression. Two things make sub-percent changes legible
here.

**An isolated core.** Core 1 and its SMT sibling (17 on this 7950X) are taken
from the kernel at boot, so nothing else runs on that physical core, the
scheduler tick is off, RCU callbacks are offloaded, and interrupts are routed to
the other cores:

```
isolcpus=domain,managed_irq,1,17 nohz_full=1,17 rcu_nocbs=1,17 irqaffinity=0,2-16,18-31
```

The benchmark then runs pinned and real-time on that core (`taskset -c 1 chrt -f
99`), with CPU boost off. Isolating the sibling matters: a thread on the other
SMT lane would share execution ports and add noise.

**A drift anchor.** Even on an isolated core, absolute throughput wanders a few
percent between runs, so comparing one decoder's MP/s before and after a change
is unreliable. The fix is to never compare across runs. Both things under
comparison are timed in the same run, microseconds apart on the same image, and
only their per-image ratio is reported. Clock, thermal, and layout drift are
common-mode: they hit both identically and cancel in the ratio. The harness does
this for ffpng against image-png; the same trick A/Bs two versions of one decoder
(register both, alternate per image, take the ratio). The anchor is strong enough
that the comparison does not even need a fixed clock: the overall ratio comes out
to three digits with CPU boost on and off, though the absolute numbers move about
20% between them. So the core isolation and boost-off above are there to steady
the absolute MP/s and tighten the median; the ratio is what makes a change
legible, down to ~0.1% of the corpus geomean during development.

**Median and minimum.** Each image is decoded for a fixed time budget over many
iterations; the harness keeps the median, which is robust to the occasional
outlier, and the minimum. External noise can only add time, never remove it, so
the minimum is the closest estimate of the true cost; the median is the reported
figure.

**Wall-clock, not instruction counts.** The bottleneck here is a latency-bound
recurrence: each Huffman table load's address depends on the previous load's
result (see The limit). Instruction-count and cachegrind-style models assume a
fixed cost per instruction and systematically misjudge such dependent-load
chains, which is why their estimated-cycle deltas drift away from real runtime.
Wall-clock on an isolated core, read through the ratio, measures what actually
moves.

## Numbers

Ryzen 7950X, single core at real-time priority, boost off, gcc 16.1, image-png
v0.18.1 `--features=unstable` at `target-cpu=x86-64-v3`, 2848 images. Both
decoders are pure AVX2. Sorted worst-first. `ffpng` and `image-png` are geomean
megapixels/second; `ratio` is `ffpng / image-png`.

The per-image ratio is independent of clock: an earlier boost-on/boost-off pair
gave the same overall ratio to three digits, since the two decoders are timed
microseconds apart on the same image and scale together. Only the fixed-clock
numbers are shown.

| category        | ffpng | image-png | ratio |    N |
|-----------------|------:|----------:|------:|-----:|
| photo_tecnick   | 209.8 |     218.5 | 0.960 |  100 |
| photo_wikipedia | 166.5 |     167.2 | 0.996 |   49 |
| textures_photo  | 158.8 |     157.3 | 1.009 |   20 |
| icon_64         | 246.9 |     240.0 | 1.029 |  213 |
| textures_pk     | 429.3 |     412.2 | 1.041 | 1002 |
| photo_kodak     | 163.4 |     155.5 | 1.051 |   24 |
| textures_plants | 275.1 |     259.2 | 1.061 |   60 |
| pngimg          | 345.7 |     323.9 | 1.067 |  187 |
| textures_pk02   | 191.8 |     179.0 | 1.071 |  235 |
| textures_pk01   | 232.9 |     214.6 | 1.085 |  113 |
| screenshot_game | 384.8 |     345.9 | 1.112 |  618 |
| icon_512        | 606.4 |     502.3 | 1.207 |  213 |
| screenshot_web  | 584.6 |     398.9 | 1.466 |   14 |
| **overall**     | **348.4** | **325.2** | **1.071** | 2848 |

The honest read: `photo_tecnick` and `photo_wikipedia` are the two categories
ffpng does not win. They are the densest-literal sets in the corpus (tecnick is
1200x1200 RGB, only ~1.8x compressible), so the decode is almost entirely the
serial Huffman literal chain, and there fdeflate is a few percent faster.
Everything else is a win, carried by the two highest-weight categories,
`textures_pk` (35% of the corpus) and `screenshot_game` (22%).

## The limit

After matching fdeflate's table layout, `photo_tecnick` is within a few percent
of image-png instead of far behind, but it does not pass it, and that is the
floor. Dense-literal data is decoded by a serial
recurrence: a symbol's position in the bitstream is known only once the previous
symbol's code length has been decoded, so the table loads form a dependent chain
of roughly one L1 latency per literal. The speculative cascade overlaps those
latencies as far as the buffered bits allow, but the load addresses are
themselves dependent, since each index needs the prior symbol's length, so the
chain cannot be flattened. fdeflate hits the same wall; matching its entry layout
closed most of the gap, and the few percent that remain are not algorithmic. The
same recurrence compiled with clang, the LLVM that also compiles fdeflate, runs a
few percent faster than with gcc, so the residual is GCC-versus-LLVM codegen on
this one loop, not a missing optimisation; ffpng ships gcc because it is about
10% faster across the rest of the corpus. The only way past the recurrence itself
is decoding more than one symbol per load (a multi-symbol table), which no
production DEFLATE decoder does and which cannot pay for its build here: the
per-block table would be 16x larger, while each dynamic block in these
photographs decodes only about 30,000 literals, far too few to amortise it.

On match-heavy data there is measured headroom: a decoder using libdeflate's
inflate as a drop-in (same front end, same unfilter) is faster than ffpng on the
match-dense categories. That gap is in libdeflate's loop structure, not any
single operation; attempts to port it either netted nothing or cost more on the
literal path than they saved. So ffpng beats image-png across the corpus, but it
is not the fastest possible inflate on every kind of data.

## Layout

```
ffpng.c  ffpng.h    the decoder (one TU; libc + immintrin only)
ldpng.c             reference decoder: libdeflate's inflate, ffpng's front end
bench.c             benchmark driver: registers decoders, times, checks output
build.sh            builds the shim, then ffpng.o + ldpng.o + bench
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
