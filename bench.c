// Copyright (c) 2026 Peter Fors
// SPDX-License-Identifier: MIT
//
// PNG decode benchmark harness.
//
// One process, one timer, one corpus walk, identical methodology for every
// decoder. Each decoder produces 8-bit RGBA (width*height*4 bytes); the first
// registered decoder is the correctness oracle and every other decoder's output
// must match it byte for byte or that image is excluded from its aggregate.
//
// Per image we time many decode iterations and keep the minimum and the median.
// Throughput is megapixels per second from the median; the corpus aggregate is
// the arithmetic average and the geometric mean of per-image MP/s, matching the
// image-rs blog's reporting.
//
// Intended to be launched pinned and real-time, e.g.:
//   taskset -c 1 chrt -f 99 ./bench ../images

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <dirent.h>
#include <sys/stat.h>

#include <png.h>

#include "ffpng.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wunused-function"
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_NO_STDIO
#include "ext/stb_image.h"
#pragma GCC diagnostic pop

// [=]===^=[ types ]===========================================================[=]

struct decode_result {
	uint8_t *pixels;	// 8-bit, width*height*channels bytes, owned by the decoder
	uint32_t width;
	uint32_t height;
	uint8_t channels;	// 1=gray 2=gray+a 3=rgb 4=rgba, post-EXPAND/STRIP_16
	size_t alloc_len;	// for decoders that need the length to free (the Rust shim)
};

struct decoder {
	char name[24];
	int  (*decode)(uint8_t *data, size_t len, struct decode_result *out);	// 0 == ok
	void (*free_pixels)(struct decode_result *r);

	// Accumulated results across the corpus.
	uint32_t passed;
	uint32_t failed;
	double sum_mpps;	// arithmetic, median-based
	double sum_log_mpps;	// for geometric mean
	double sum_mpps_best;	// arithmetic, min-time-based
};

struct mem_reader {
	uint8_t *data;
	size_t len;
	size_t pos;
};

// [=]===^=[ globals ]=========================================================[=]

static uint64_t cfg_min_time_ns = 50000000;	// 50 ms accumulated per image
static uint32_t cfg_min_iters = 5;
static uint32_t cfg_max_iters = 2000;
static uint32_t cfg_limit = 0;			// 0 == no limit
static char *cfg_filter = 0;			// substring match on path, or null
static char *cfg_csv = 0;
static char *cfg_only = 0;			// run only this decoder name, or null

static char **paths = 0;
static uint32_t path_count = 0;
static uint32_t path_cap = 0;

// [=]===^=[ now_ns ]==========================================================[=]
static uint64_t now_ns(void) {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000u + (uint64_t)ts.tv_nsec;
}

// [=]===^=[ cmp_u64 ]=========================================================[=]
static int cmp_u64(const void *a, const void *b) {
	uint64_t x = *(const uint64_t *)a;
	uint64_t y = *(const uint64_t *)b;
	return (x > y) - (x < y);
}

// [=]===^=[ cmp_str ]=========================================================[=]
static int cmp_str(const void *a, const void *b) {
	return strcmp(*(char *const *)a, *(char *const *)b);
}

// [=]===^=[ read_file ]=======================================================[=]
static uint8_t *read_file(char *path, size_t *out_len) {
	FILE *f = fopen(path, "rb");
	if(!f) {
		return 0;
	}
	fseek(f, 0, SEEK_END);
	long n = ftell(f);
	fseek(f, 0, SEEK_SET);
	uint8_t *buf = malloc((size_t)n);
	if(buf && fread(buf, 1, (size_t)n, f) != (size_t)n) {
		free(buf);
		buf = 0;
	}
	fclose(f);
	*out_len = (size_t)n;
	return buf;
}

// [=]===^=[ parse_ihdr ]======================================================[=]
// Pulls width/height straight from IHDR so the harness can size the comparison
// buffers without trusting any one decoder. Returns 0 on success.
static int parse_ihdr(uint8_t *data, size_t len, uint32_t *w, uint32_t *h) {
	static const uint8_t sig[8] = { 0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a };
	if(len < 33 || memcmp(data, sig, 8) != 0 || memcmp(data + 12, "IHDR", 4) != 0) {
		return 1;
	}
	*w = (uint32_t)data[16] << 24 | (uint32_t)data[17] << 16 | (uint32_t)data[18] << 8 | data[19];
	*h = (uint32_t)data[20] << 24 | (uint32_t)data[21] << 16 | (uint32_t)data[22] << 8 | data[23];
	return 0;
}

// [=]===^=[ stb_decode ]======================================================[=]
static int stb_decode(uint8_t *data, size_t len, struct decode_result *out) {
	int w, h, ch;
	uint8_t *p = stbi_load_from_memory(data, (int)len, &w, &h, &ch, 0);
	if(!p) {
		return 1;
	}
	out->pixels = p;
	out->width = (uint32_t)w;
	out->height = (uint32_t)h;
	out->channels = (uint8_t)ch;
	return 0;
}

// [=]===^=[ stb_free ]========================================================[=]
static void stb_free(struct decode_result *r) {
	stbi_image_free(r->pixels);
}

// [=]===^=[ png_mem_read ]====================================================[=]
static void png_mem_read(png_structp png, png_bytep dst, png_size_t want) {
	struct mem_reader *r = png_get_io_ptr(png);
	size_t avail = r->len - r->pos;
	size_t n = want < avail ? want : avail;
	memcpy(dst, r->data + r->pos, n);
	r->pos += n;
	if(n < want) {
		png_error(png, "short read");
	}
}

// [=]===^=[ libpng_decode ]===================================================[=]
static int libpng_decode(uint8_t *data, size_t len, struct decode_result *out) {
	png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, 0, 0, 0);
	if(!png) {
		return 1;
	}
	png_infop info = png_create_info_struct(png);
	if(!info) {
		png_destroy_read_struct(&png, 0, 0);
		return 1;
	}

	uint8_t *image = 0;
	png_bytep *rows = 0;
	if(setjmp(png_jmpbuf(png))) {
		free(image);
		free(rows);
		png_destroy_read_struct(&png, &info, 0);
		return 1;
	}

	struct mem_reader r = { data, len, 0 };
	png_set_read_fn(png, &r, png_mem_read);
	png_read_info(png, info);

	uint32_t w = png_get_image_width(png, info);
	uint32_t h = png_get_image_height(png, info);
	int bd = png_get_bit_depth(png, info);
	int ct = png_get_color_type(png, info);

	// EXPAND | STRIP_16 semantics (the image crate default): palette to RGB,
	// sub-8-bit gray to 8-bit, tRNS to alpha, 16-bit to 8-bit. No forced alpha
	// and no gray-to-RGB, so channel count stays native.
	if(ct == PNG_COLOR_TYPE_PALETTE) {
		png_set_palette_to_rgb(png);
	}
	if(ct == PNG_COLOR_TYPE_GRAY && bd < 8) {
		png_set_expand_gray_1_2_4_to_8(png);
	}
	if(png_get_valid(png, info, PNG_INFO_tRNS)) {
		png_set_tRNS_to_alpha(png);
	}
	if(bd == 16) {
		png_set_strip_16(png);
	}
	png_set_interlace_handling(png);
	png_read_update_info(png, info);

	uint32_t channels = png_get_channels(png, info);
	size_t rowbytes = png_get_rowbytes(png, info);
	image = malloc((size_t)h * rowbytes);
	rows = malloc((size_t)h * sizeof(png_bytep));
	if(!image || !rows) {
		png_error(png, "oom");
	}
	for(uint32_t y = 0; y < h; ++y) {
		rows[y] = image + (size_t)y * rowbytes;
	}
	png_read_image(png, rows);

	free(rows);
	png_destroy_read_struct(&png, &info, 0);
	out->pixels = image;
	out->width = w;
	out->height = h;
	out->channels = (uint8_t)channels;
	return 0;
}

// [=]===^=[ libpng_free ]=====================================================[=]
static void libpng_free(struct decode_result *r) {
	free(r->pixels);
}

// [=]===^=[ imagepng (rust shim) ]============================================[=]
struct shim_result {
	uint8_t *pixels;
	size_t len;
	uint32_t width;
	uint32_t height;
	uint8_t channels;
};
extern int imagepng_decode(uint8_t *data, size_t len, struct shim_result *out);
extern void imagepng_free(uint8_t *pixels, size_t len);

static int imagepng_dec(uint8_t *data, size_t len, struct decode_result *out) {
	struct shim_result r;
	if(imagepng_decode(data, len, &r) != 0) {
		return 1;
	}
	out->pixels = r.pixels;
	out->width = r.width;
	out->height = r.height;
	out->channels = r.channels;
	out->alloc_len = r.len;
	return 0;
}

// [=]===^=[ imagepng_free_result ]============================================[=]
static void imagepng_free_result(struct decode_result *r) {
	imagepng_free(r->pixels, r->alloc_len);
}

// [=]===^=[ to_rgba8 ]========================================================[=]
// Expands a native 8-bit decode to canonical RGBA8 for byte-exact comparison.
// Correctness only, never on the timed path.
static void to_rgba8(uint8_t *src, uint32_t w, uint32_t h, uint8_t channels, uint8_t *dst) {
	size_t n = (size_t)w * h;
	switch(channels) {
		case 1: {
			for(size_t i = 0; i < n; ++i) {
				dst[i * 4 + 0] = src[i];
				dst[i * 4 + 1] = src[i];
				dst[i * 4 + 2] = src[i];
				dst[i * 4 + 3] = 0xff;
			}
			break;
		}

		case 2: {
			for(size_t i = 0; i < n; ++i) {
				dst[i * 4 + 0] = src[i * 2];
				dst[i * 4 + 1] = src[i * 2];
				dst[i * 4 + 2] = src[i * 2];
				dst[i * 4 + 3] = src[i * 2 + 1];
			}
			break;
		}

		case 3: {
			for(size_t i = 0; i < n; ++i) {
				dst[i * 4 + 0] = src[i * 3 + 0];
				dst[i * 4 + 1] = src[i * 3 + 1];
				dst[i * 4 + 2] = src[i * 3 + 2];
				dst[i * 4 + 3] = 0xff;
			}
			break;
		}

		case 4: {
			memcpy(dst, src, n * 4);
			break;
		}
	}
}

// [=]===^=[ ffpng ]============================================================[=]
static int ffpng_decode(uint8_t *data, size_t len, struct decode_result *out) {
	struct pd_image img;
	if(pd_decode(data, len, &img) != 0) {
		return 1;
	}
	out->pixels = img.pixels;
	out->width = img.width;
	out->height = img.height;
	out->channels = img.channels;
	return 0;
}

// [=]===^=[ ffpng_free ]=======================================================[=]
static void ffpng_free(struct decode_result *r) {
	free(r->pixels);
}

// [=]===^=[ decoders ]========================================================[=]
// The first decoder is the correctness oracle; image-png defines the EXPAND
// output we must reproduce.
static struct decoder decoders[] = {
	{ .name = "image-png", .decode = imagepng_dec, .free_pixels = imagepng_free_result },
	{ .name = "ffpng", .decode = ffpng_decode, .free_pixels = ffpng_free },
	{ .name = "stb_image", .decode = stb_decode, .free_pixels = stb_free },
	{ .name = "libpng", .decode = libpng_decode, .free_pixels = libpng_free },
};
static const uint32_t decoder_count = sizeof(decoders) / sizeof(decoders[0]);

// [=]===^=[ collect_paths ]===================================================[=]
static void collect_paths(char *dir) {
	DIR *d = opendir(dir);
	if(!d) {
		return;
	}
	struct dirent *e;
	while((e = readdir(d))) {
		if(e->d_name[0] == '.') {
			continue;
		}
		char path[4096];
		snprintf(path, sizeof(path), "%s/%s", dir, e->d_name);
		struct stat st;
		if(stat(path, &st) != 0) {
			continue;
		}
		if(S_ISDIR(st.st_mode)) {
			collect_paths(path);
		} else {
			size_t n = strlen(e->d_name);
			if(n < 4 || strcmp(e->d_name + n - 4, ".png") != 0) {
				continue;
			}
			if(cfg_filter && !strstr(path, cfg_filter)) {
				continue;
			}
			if(path_count == path_cap) {
				path_cap = path_cap ? path_cap * 2 : 1024;
				paths = realloc(paths, path_cap * sizeof(char *));
			}
			paths[path_count++] = strdup(path);
		}
	}
	closedir(d);
}

// [=]===^=[ bench_image ]=====================================================[=]
// Times every decoder on one already-loaded PNG. The first decoder's output is
// the reference; mismatches are reported and excluded from the aggregate.
static void bench_image(char *path, uint8_t *data, size_t len, uint32_t w, uint32_t h, uint64_t *iter_buf, FILE *csv) {
	size_t expect = (size_t)w * h * 4;
	uint8_t *ref_rgba = malloc(expect);
	uint8_t *cand_rgba = malloc(expect);

	struct decode_result ref;
	if(decoders[0].decode(data, len, &ref) != 0 || ref.width != w || ref.height != h) {
		printf("  ORACLE FAIL %s\n", path);
		free(ref_rgba);
		free(cand_rgba);
		return;
	}
	to_rgba8(ref.pixels, w, h, ref.channels, ref_rgba);
	decoders[0].free_pixels(&ref);

	for(uint32_t di = 0; di < decoder_count; ++di) {
		struct decoder *dec = &decoders[di];
		if(cfg_only && strcmp(cfg_only, dec->name) != 0) {
			continue;
		}

		struct decode_result chk;
		if(dec->decode(data, len, &chk) != 0) {
			printf("  %-10s DECODE FAIL %s\n", dec->name, path);
			dec->failed++;
			continue;
		}
		int ok = chk.width == w && chk.height == h;
		if(ok) {
			to_rgba8(chk.pixels, w, h, chk.channels, cand_rgba);
			ok = memcmp(cand_rgba, ref_rgba, expect) == 0;
		}
		dec->free_pixels(&chk);
		if(!ok) {
			printf("  %-10s MISMATCH %s\n", dec->name, path);
			dec->failed++;
			continue;
		}

		uint32_t iters = 0;
		uint64_t start = now_ns();
		while(iters < cfg_max_iters) {
			uint64_t t0 = now_ns();
			struct decode_result tr;
			dec->decode(data, len, &tr);
			uint64_t dt = now_ns() - t0;
			dec->free_pixels(&tr);
			iter_buf[iters++] = dt;
			if(iters >= cfg_min_iters && now_ns() - start >= cfg_min_time_ns) {
				break;
			}
		}

		qsort(iter_buf, iters, sizeof(uint64_t), cmp_u64);
		uint64_t med = iter_buf[iters / 2];
		uint64_t best = iter_buf[0];
		double px = (double)w * h;
		double mpps_med = px / ((double)med / 1e9) / 1e6;
		double mpps_best = px / ((double)best / 1e9) / 1e6;

		dec->passed++;
		dec->sum_mpps += mpps_med;
		dec->sum_log_mpps += log(mpps_med);
		dec->sum_mpps_best += mpps_best;

		if(csv) {
			fprintf(csv, "%s,%s,%u,%u,%llu,%llu,%.3f\n", path, dec->name, w, h, (unsigned long long)med, (unsigned long long)best, mpps_med);
		}
	}

	free(ref_rgba);
	free(cand_rgba);
}

// [=]===^=[ report ]==========================================================[=]
static void report(void) {
	printf("\n%-12s %8s %8s   %12s %12s\n", "decoder", "passed", "failed", "avg MP/s", "geomean");
	printf("-----------------------------------------------------------------\n");
	for(uint32_t di = 0; di < decoder_count; ++di) {
		struct decoder *dec = &decoders[di];
		if(cfg_only && strcmp(cfg_only, dec->name) != 0) {
			continue;
		}
		double avg = dec->passed ? dec->sum_mpps / dec->passed : 0.0;
		double geo = dec->passed ? exp(dec->sum_log_mpps / dec->passed) : 0.0;
		printf("%-12s %8u %8u   %12.1f %12.1f\n", dec->name, dec->passed, dec->failed, avg, geo);
	}
}

// [=]===^=[ main ]============================================================[=]
int main(int argc, char **argv) {
	char *dir = 0;
	for(int i = 1; i < argc; ++i) {
		if(strcmp(argv[i], "--min-time-ms") == 0 && i + 1 < argc) {
			cfg_min_time_ns = (uint64_t)strtoull(argv[++i], 0, 10) * 1000000u;
		} else if(strcmp(argv[i], "--min-iters") == 0 && i + 1 < argc) {
			cfg_min_iters = (uint32_t)strtoul(argv[++i], 0, 10);
		} else if(strcmp(argv[i], "--max-iters") == 0 && i + 1 < argc) {
			cfg_max_iters = (uint32_t)strtoul(argv[++i], 0, 10);
		} else if(strcmp(argv[i], "--limit") == 0 && i + 1 < argc) {
			cfg_limit = (uint32_t)strtoul(argv[++i], 0, 10);
		} else if(strcmp(argv[i], "--filter") == 0 && i + 1 < argc) {
			cfg_filter = argv[++i];
		} else if(strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
			cfg_csv = argv[++i];
		} else if(strcmp(argv[i], "--decoder") == 0 && i + 1 < argc) {
			cfg_only = argv[++i];
		} else {
			dir = argv[i];
		}
	}
	if(!dir) {
		printf("usage: bench [options] <corpus_dir>\n");
		return 1;
	}

	collect_paths(dir);
	qsort(paths, path_count, sizeof(char *), cmp_str);
	if(cfg_limit && cfg_limit < path_count) {
		path_count = cfg_limit;
	}
	printf("corpus: %u images from %s\n", path_count, dir);

	FILE *csv = 0;
	if(cfg_csv) {
		csv = fopen(cfg_csv, "wb");
		if(csv) {
			fprintf(csv, "path,decoder,width,height,median_ns,min_ns,mpps_median\n");
		}
	}

	uint64_t *iter_buf = malloc(cfg_max_iters * sizeof(uint64_t));
	for(uint32_t i = 0; i < path_count; ++i) {
		size_t len;
		uint8_t *data = read_file(paths[i], &len);
		if(!data) {
			continue;
		}
		uint32_t w, h;
		if(parse_ihdr(data, len, &w, &h) == 0 && w && h) {
			bench_image(paths[i], data, len, w, h, iter_buf, csv);
		}
		free(data);
	}
	free(iter_buf);

	if(csv) {
		fclose(csv);
	}
	report();
	return 0;
}
