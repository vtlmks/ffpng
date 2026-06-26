// Copyright (c) 2026 Peter Fors
// SPDX-License-Identifier: MIT
//
// libdeflate-backed PNG decoder used ONLY as a benchmark reference. It reuses
// ffpng's exact chunk parse, unfilter and expand by including ffpng.c, and
// swaps the one component under test: the inflate engine becomes a single
// libdeflate_zlib_decompress call instead of ffpng's own inflate_stream. So a
// head-to-head ldpng-vs-ffpng number isolates the inflate engine and nothing
// else (identical CRC, unfilter, palette/expand code paths).
//
// ffpng.c's two public symbols are macro-renamed before the include so this
// translation unit does not collide with the real ffpng.o at link time; the
// renamed copies are unreferenced and dropped by --gc-sections.

#include <libdeflate.h>

#define pd_decode pd_decode_ldunused
#define pd_free   pd_free_ldunused
#include "ffpng.c"
#undef pd_decode
#undef pd_free

// [=]===^=[ globals ]=========================================================[=]

static struct libdeflate_decompressor *ld_decompressor;

static uint8_t *ld_raw;
static size_t ld_raw_cap;

// [=]===^=[ ld_raw_grow ]=====================================================[=]
// Grow-only scratch for the filtered raw bytes, mirroring ffpng's raw_scratch
// so neither decoder pays a per-decode malloc on large images.
static uint8_t *ld_raw_grow(size_t n) {
	if(n + COPY_PAD > ld_raw_cap) {
		free(ld_raw);
		ld_raw_cap = n + COPY_PAD;
		ld_raw = malloc(ld_raw_cap);
	}
	return ld_raw;
}

// [=]===^=[ ldpng_decode ]====================================================[=]
// Same as ffpng's pd_decode but inflate is libdeflate. Chunk parse and the
// unfilter/expand orchestration are deliberately duplicated here (this is a
// distinct decoder), calling into the shared static helpers from ffpng.c.
int ldpng_decode(uint8_t *data, size_t len, struct pd_image *out) {
	static const uint8_t sig[8] = { 0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a };
	if(!crc_ready) {
		crc_build();
	}
	if(!ld_decompressor) {
		ld_decompressor = libdeflate_alloc_decompressor();
	}
	if(!ld_decompressor || len < 8 || memcmp(data, sig, 8) != 0) {
		return 1;
	}

	uint32_t width = 0, height = 0;
	uint32_t bd = 0;
	int ct = -1;
	int interlace = 0;
	uint8_t plte[256 * 3];
	uint8_t trns[256];
	uint32_t trns_len = 0;
	int have_ihdr = 0;

	uint8_t *idat = 0;
	size_t idat_len = 0, idat_cap = 0;

	size_t pos = 8;
	int saw_iend = 0;
	while(pos + 8 <= len) {
		uint32_t clen = (uint32_t)data[pos] << 24 | (uint32_t)data[pos + 1] << 16 | (uint32_t)data[pos + 2] << 8 | data[pos + 3];
		uint8_t *type = data + pos + 4;
		if(pos + 12 + clen > len) {
			break;
		}
		uint8_t *cdata = data + pos + 8;
		if(!(type[0] & 0x20)) {
			uint32_t stored = (uint32_t)data[pos + 8 + clen] << 24 | (uint32_t)data[pos + 9 + clen] << 16 | (uint32_t)data[pos + 10 + clen] << 8 | data[pos + 11 + clen];
			if(crc32_calc(type, clen + 4) != stored) {
				free(idat);
				return 1;
			}
		}

		if(memcmp(type, "IHDR", 4) == 0) {
			if(clen != 13) {
				free(idat);
				return 1;
			}
			width = (uint32_t)cdata[0] << 24 | (uint32_t)cdata[1] << 16 | (uint32_t)cdata[2] << 8 | cdata[3];
			height = (uint32_t)cdata[4] << 24 | (uint32_t)cdata[5] << 16 | (uint32_t)cdata[6] << 8 | cdata[7];
			bd = cdata[8];
			ct = cdata[9];
			interlace = cdata[12];
			have_ihdr = 1;
		} else if(memcmp(type, "PLTE", 4) == 0) {
			if(clen > sizeof(plte)) {
				free(idat);
				return 1;
			}
			memcpy(plte, cdata, clen);
		} else if(memcmp(type, "tRNS", 4) == 0) {
			trns_len = clen > sizeof(trns) ? sizeof(trns) : clen;
			memcpy(trns, cdata, trns_len);
		} else if(memcmp(type, "IDAT", 4) == 0) {
			if(idat_len + clen > idat_cap) {
				idat_cap = (idat_len + clen) * 2;
				uint8_t *grown = realloc(idat, idat_cap);
				if(!grown) {
					free(idat);
					return 1;
				}
				idat = grown;
			}
			memcpy(idat + idat_len, cdata, clen);
			idat_len += clen;
		} else if(memcmp(type, "IEND", 4) == 0) {
			saw_iend = 1;
			break;
		}
		pos += 12 + clen;
	}

	if(!have_ihdr || !saw_iend || width == 0 || height == 0) {
		free(idat);
		return 1;
	}
	if(interlace != 0 && interlace != 1) {
		free(idat);
		return 1;
	}

	uint32_t rch = raw_channels(ct);
	uint32_t out_ch = output_channels(ct, (ct == 0 || ct == 2 || ct == 3) ? trns_len : 0);
	if(rch == 0 || out_ch == 0) {
		free(idat);
		return 1;
	}
	uint32_t eff_trns = (ct == 0 || ct == 2 || ct == 3) ? trns_len : 0;
	size_t bits_per_pixel = (size_t)rch * bd;
	uint32_t bpp = (uint32_t)((bits_per_pixel + 7) / 8);
	if(bpp == 0) {
		bpp = 1;
	}

	uint32_t pal32[256];
	uint32_t *pal32_ptr = 0;
	if(ct == 3 && bd == 8) {
		for(uint32_t i = 0; i < 256; ++i) {
			uint32_t a = i < eff_trns ? trns[i] : 0xff;
			pal32[i] = plte[i * 3 + 0] | (uint32_t)plte[i * 3 + 1] << 8 | (uint32_t)plte[i * 3 + 2] << 16 | a << 24;
		}
		pal32_ptr = pal32;
	}

	uint8_t *pixels = malloc((size_t)width * height * out_ch);
	if(!pixels) {
		free(idat);
		return 1;
	}

	if(interlace == 0) {
		size_t row_bytes = (width * bits_per_pixel + 7) / 8;
		size_t stride = row_bytes + 1;
		size_t raw_size = stride * height;
		uint8_t *raw = ld_raw_grow(raw_size);
		size_t got = 0;
		// Strip the 2-byte zlib header and run raw DEFLATE so libdeflate does NOT
		// validate adler32 over the output, matching ffpng (which ignores the
		// trailer). This keeps the comparison purely the inflate engine.
		if(!raw || idat_len < 2 || libdeflate_deflate_decompress(ld_decompressor, idat + 2, idat_len - 2, raw, raw_size, &got) != 0 || got != raw_size) {
			free(idat);
			free(pixels);
			return 1;
		}
		uint8_t *prev = 0;
		for(uint32_t y = 0; y < height; ++y) {
			uint8_t *cur = raw + (size_t)y * stride + 1;
			if(unfilter_row(cur, prev, raw[(size_t)y * stride], row_bytes, bpp) != 0) {
				free(idat);
				free(pixels);
				return 1;
			}
			expand_row(cur, pixels + (size_t)y * width * out_ch, width, bd, ct, plte, trns, eff_trns, out_ch, pal32_ptr);
			prev = cur;
		}
	} else {
		uint32_t pw[7], ph[7];
		size_t prb[7], poff[7];
		size_t raw_size = 0;
		uint32_t maxw = 0;
		for(uint32_t p = 0; p < 7; ++p) {
			pw[p] = width > a7_xorig[p] ? (width - a7_xorig[p] + a7_xstep[p] - 1) / a7_xstep[p] : 0;
			ph[p] = height > a7_yorig[p] ? (height - a7_yorig[p] + a7_ystep[p] - 1) / a7_ystep[p] : 0;
			prb[p] = (pw[p] * bits_per_pixel + 7) / 8;
			poff[p] = raw_size;
			if(pw[p] && ph[p]) {
				raw_size += (prb[p] + 1) * (size_t)ph[p];
			}
			if(pw[p] > maxw) {
				maxw = pw[p];
			}
		}
		uint8_t *raw = ld_raw_grow(raw_size);
		uint8_t *temp = malloc((size_t)maxw * out_ch);
		size_t got = 0;
		if(!raw || !temp || idat_len < 2 || libdeflate_deflate_decompress(ld_decompressor, idat + 2, idat_len - 2, raw, raw_size, &got) != 0 || got != raw_size) {
			free(temp);
			free(idat);
			free(pixels);
			return 1;
		}
		int ok = 1;
		for(uint32_t p = 0; p < 7; ++p) {
			if(!pw[p] || !ph[p]) {
				continue;
			}
			uint8_t *praw = raw + poff[p];
			if(unfilter(praw, ph[p], prb[p], bpp) != 0) {
				ok = 0;
				break;
			}
			for(uint32_t r = 0; r < ph[p]; ++r) {
				expand_row(praw + (size_t)r * (prb[p] + 1) + 1, temp, pw[p], bd, ct, plte, trns, eff_trns, out_ch, pal32_ptr);
				uint32_t y = a7_yorig[p] + r * a7_ystep[p];
				uint8_t *dstrow = pixels + (size_t)y * width * out_ch;
				for(uint32_t c = 0; c < pw[p]; ++c) {
					uint32_t x = a7_xorig[p] + c * a7_xstep[p];
					memcpy(dstrow + (size_t)x * out_ch, temp + (size_t)c * out_ch, out_ch);
				}
			}
		}
		free(temp);
		if(!ok) {
			free(idat);
			free(pixels);
			return 1;
		}
	}
	free(idat);

	out->pixels = pixels;
	out->width = width;
	out->height = height;
	out->channels = (uint8_t)out_ch;
	return 0;
}
