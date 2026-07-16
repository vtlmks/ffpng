// Copyright (c) 2026 Peter Fors
// SPDX-License-Identifier: MIT

#include "ffpng.h"
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

// [=]===^=[ types ]===========================================================[=]

struct bitreader {
	uint8_t *in;
	uint8_t *end;
	uint64_t bitbuf;
	uint32_t bitcnt;
};

#define FAST_BITS 9
#define FAST12_BITS 12
// comb packs code length in 0..7, literal advance in 8..9, pair length in 10..13, EOB/LIT in 14/15, payload in 16..31; zero selects canonical decoding.
#define COMB_LIT 0x8000u
#define COMB_EOB 0x4000u
struct huff {
	uint16_t fast[1 << FAST_BITS];
	uint32_t comb[1 << FAST12_BITS];
	uint32_t dist_comb[1 << FAST_BITS];
	uint16_t firstcode[16];
	int32_t maxcode[17];
	uint16_t firstsymbol[16];
	uint8_t size[288];
	uint16_t value[288];
};

// Rows beyond the DEFLATE window are safe to mutate and consume while still resident in L2.
#define INFLATE_WINDOW 32768

// 128 KB amortizes comb-table reloads; 8 KB regressed moderate inflate-bound images.
#define STREAM_CHUNK (128 * 1024)

// Streaming wins beyond LLC capacity; below 16 MB the L3 reread is cheaper and interleaving costs about 0.3%.
#define STREAM_MIN (16u << 20)

struct sink {
	uint8_t *raw;
	uint8_t *pixels;
	uint8_t *prev;
	uint8_t *plte;
	uint8_t *trns;
	uint32_t *pal32;
	size_t stride;
	size_t row_bytes;
	uint32_t crow;
	uint32_t height;
	uint32_t width;
	uint32_t bpp;
	uint32_t bd;
	uint32_t out_ch;
	uint32_t eff_trns;
	int32_t ct;
};

// [=]===^=[ globals ]=========================================================[=]

static uint16_t length_base[29] = {
	3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258
};

static uint8_t length_extra[29] = {
	0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0
};

static uint16_t dist_base[30] = {
	1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577
};

static uint8_t dist_extra[30] = {
	0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13
};

static uint8_t clcidx[19] = {
	16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15
};

static uint8_t a7_xorig[7] = { 0, 4, 0, 2, 0, 1, 0 };
static uint8_t a7_yorig[7] = { 0, 0, 4, 0, 2, 0, 1 };
static uint8_t a7_xstep[7] = { 8, 8, 4, 4, 2, 2, 1 };
static uint8_t a7_ystep[7] = { 8, 8, 8, 4, 4, 2, 2 };

static uint32_t crc_table[256];
static uint32_t crc_ready;

// Reuse avoids page faults above this threshold; smaller fixed-address buffers regress through 4K aliasing.
#define RAW_REUSE_MIN (256 * 1024)
// 32 KB regressed textures_pk from 1.184 to 1.158 because match-heavy palette data cannot amortize comb construction.
#define BIG_MIN 131072
// Padding makes unconditional vector stores safe; slop is overwritten before it can become output.
#define COPY_PAD 16
static uint8_t *raw_scratch;
static size_t raw_scratch_cap;

// [=]===^=[ raw_grow ]========================================================[=]
static uint8_t *raw_grow(size_t n) {
	if(n > raw_scratch_cap) {
		free(raw_scratch);
		raw_scratch = malloc(n + COPY_PAD);
		raw_scratch_cap = raw_scratch ? n : 0;
	}
	return raw_scratch;
}

// [=]===^=[ crc_build ]=======================================================[=]
static void crc_build(void) {
	for(uint32_t n = 0; n < 256; ++n) {
		uint32_t c = n;
		for(uint32_t k = 0; k < 8; ++k) {
			c = (c & 1) ? 0xedb88320u ^ (c >> 1) : c >> 1;
		}
		crc_table[n] = c;
	}
	crc_ready = 1;
}

// [=]===^=[ crc_bytes ]=======================================================[=]
static uint32_t crc_bytes(uint32_t c, uint8_t *buf, size_t len) {
	while(len--) {
		c = crc_table[(c ^ *buf++) & 0xff] ^ (c >> 8);
	}
	return c;
}

// [=]===^=[ crc32_calc ]======================================================[=]
static uint32_t crc32_calc(uint8_t *buf, size_t len) {
	if(len < 16) {
		return crc_bytes(0xffffffffu, buf, len) ^ 0xffffffffu;
	}
	size_t bulk = len & ~(size_t)15;
	__m128i k34 = _mm_set_epi64x(0xccaa009eLL, 0x1751997d0LL);
	__m128i x1 = _mm_loadu_si128((__m128i *)buf);
	x1 = _mm_xor_si128(x1, _mm_cvtsi32_si128((int)0xffffffffu));
	for(size_t o = 16; o < bulk; o += 16) {
		__m128i x2 = _mm_loadu_si128((__m128i *)(buf + o));
		__m128i x5 = _mm_clmulepi64_si128(x1, k34, 0x00);
		x1 = _mm_clmulepi64_si128(x1, k34, 0x11);
		x1 = _mm_xor_si128(_mm_xor_si128(x1, x2), x5);
	}

	// Reflected 128 -> 32 reduction (verified against the bit-serial CRC).
	__m128i k50 = _mm_set_epi64x(0, 0x163cd6124LL);
	__m128i poly = _mm_set_epi64x(0x1f7011641LL, 0x1db710641LL);	// [mu : P]
	__m128i m32 = _mm_set_epi32(0, (int)0xffffffff, 0, (int)0xffffffff);
	__m128i x2 = _mm_clmulepi64_si128(x1, k34, 0x10);
	x1 = _mm_xor_si128(_mm_srli_si128(x1, 8), x2);
	x2 = _mm_srli_si128(x1, 4);
	x1 = _mm_and_si128(x1, m32);
	x1 = _mm_clmulepi64_si128(x1, k50, 0x00);
	x1 = _mm_xor_si128(x1, x2);
	x2 = _mm_and_si128(x1, m32);
	x2 = _mm_clmulepi64_si128(x2, poly, 0x10);
	x2 = _mm_and_si128(x2, m32);
	x2 = _mm_clmulepi64_si128(x2, poly, 0x00);
	x1 = _mm_xor_si128(x1, x2);

	uint32_t crc = (uint32_t)_mm_extract_epi32(x1, 1);
	return crc_bytes(crc, buf + bulk, len - bulk) ^ 0xffffffffu;
}

// [=]===^=[ refill ]==========================================================[=]
static inline void refill(struct bitreader *b) {
	if(b->in + 8 <= b->end) {
		uint64_t chunk;
		memcpy(&chunk, b->in, 8);
		b->bitbuf |= chunk << b->bitcnt;
		uint32_t adv = (63 - b->bitcnt) >> 3;
		b->in += adv;
		b->bitcnt += adv << 3;

	} else {
		while(b->bitcnt <= 56 && b->in < b->end) {
			b->bitbuf |= (uint64_t)*b->in++ << b->bitcnt;
			b->bitcnt += 8;
		}
	}
}

// [=]===^=[ getbits ]=========================================================[=]
static inline uint32_t getbits(struct bitreader *b, uint32_t n) {
	if(b->bitcnt < n) {
		refill(b);
	}
	uint32_t v = (uint32_t)(b->bitbuf & (((uint64_t)1 << n) - 1));
	b->bitbuf >>= n;
	b->bitcnt -= n;
	return v;
}

// [=]===^=[ bit_reverse16 ]===================================================[=]
static inline uint32_t bit_reverse16(uint32_t v, uint32_t bits) {
	v = ((v & 0xaaaa) >> 1) | ((v & 0x5555) << 1);
	v = ((v & 0xcccc) >> 2) | ((v & 0x3333) << 2);
	v = ((v & 0xf0f0) >> 4) | ((v & 0x0f0f) << 4);
	v = ((v & 0xff00) >> 8) | ((v & 0x00ff) << 8);
	return v >> (16 - bits);
}

// [=]===^=[ huff_build ]======================================================[=]
// mode is 0 for literal, 1 for combined literal, 2 for combined distance, and 3 for packed distance.
static int32_t huff_build(struct huff *h, uint8_t *lengths, uint32_t num, uint32_t mode) {
	uint32_t sizes[17] = { 0 };
	uint32_t next_code[16];
	if(mode != 1) {
		memset(h->fast, 0, sizeof(h->fast));
	}
	for(uint32_t i = 0; i < num; ++i) {
		sizes[lengths[i]]++;
	}
	sizes[0] = 0;
	for(uint32_t i = 1; i < 16; ++i) {
		if(sizes[i] > (1u << i)) {
			return -1;
		}
	}

	uint32_t code = 0, k = 0;
	for(uint32_t i = 1; i < 16; ++i) {
		next_code[i] = code;
		h->firstcode[i] = (uint16_t)code;
		h->firstsymbol[i] = (uint16_t)k;
		code += sizes[i];
		if(sizes[i] && code - 1 >= (1u << i)) {
			return -1;
		}
		h->maxcode[i] = (int32_t)(code << (16 - i));
		code <<= 1;
		k += sizes[i];
	}
	h->maxcode[16] = 0x10000;

	for(uint32_t i = 0; i < num; ++i) {
		uint32_t s = lengths[i];
		if(s) {
			uint32_t c = next_code[s] - h->firstcode[s] + h->firstsymbol[s];
			h->size[c] = (uint8_t)s;
			h->value[c] = (uint16_t)i;
			// mode 1 never reads fast, so its strided scatter is dead work.
			if(mode != 1 && s <= FAST_BITS) {
				uint32_t j = bit_reverse16(next_code[s], s);
				while(j < (1u << FAST_BITS)) {
					if(mode == 3) {
						h->fast[j] = (uint16_t)(s | ((uint32_t)dist_extra[i] << 4) | (i << 8));

					} else {
						h->fast[j] = (uint16_t)((s << 9) | i);
					}
					j += 1u << s;
				}
			}
			next_code[s]++;
		}
	}

	// Table doubling is about 7x cheaper than strided scatter plus a full fusion scan while producing the same extensions.
	if(mode == 1) {
		uint16_t sorted[288];
		uint16_t codes[288];
		uint32_t offset[16];
		uint32_t o = 0;
		for(uint32_t len = 1; len < 16; ++len) {
			offset[len] = o;
			o += sizes[len];
		}
		uint32_t fill[16];
		memcpy(fill, offset, sizeof(fill));
		// Symbol-order sorting places each length group's literals first, letting fusion skip symbol tests.
		uint32_t lit_sizes[16] = { 0 };
		for(uint32_t i = 0; i < num; ++i) {
			uint32_t s = lengths[i];
			if(s) {
				sorted[fill[s]++] = (uint16_t)i;
				if(i < 256) {
					++lit_sizes[s];
				}
			}
		}
		memset(h->comb, 0, sizeof(h->comb));
		uint32_t si = 0;
		for(uint32_t len = 1; len <= FAST12_BITS; ++len) {
			uint32_t cw = h->firstcode[len];
			for(uint32_t n = 0; n < sizes[len]; ++n) {
				uint32_t i = sorted[si++];
				uint32_t idx = bit_reverse16(cw, len);
				uint32_t centry;
				if(i < 256) {
					centry = COMB_LIT | (1u << 8) | ((uint32_t)i << 16) | len;

				} else if(i == 256) {
					centry = COMB_EOB | len;

				} else if(i - 257 < 29) {
					uint32_t li = i - 257;
					// Bits 25..29 preserve the length symbol for the cold tail without a canonical lookup.
					centry = ((uint32_t)length_extra[li] << 8) | ((uint32_t)length_base[li] << 16) | (li << 25) | len;

				} else {
					centry = 0;
				}
				h->comb[idx] = centry;
				codes[i] = (uint16_t)idx;
				++cw;
			}
			// Enumerating actual literal pairs avoids scanning every comb slot.
			for(uint32_t len1 = 1; len1 < len; ++len1) {
				uint32_t len2 = len - len1;
				uint32_t end1 = offset[len1] + lit_sizes[len1];
				uint32_t end2 = offset[len2] + lit_sizes[len2];
				for(uint32_t a = offset[len1]; a < end1; ++a) {
					uint32_t sym1 = sorted[a];
					for(uint32_t b = offset[len2]; b < end2; ++b) {
						uint32_t sym2 = sorted[b];
						uint32_t idx = codes[sym1] | ((uint32_t)codes[sym2] << len1);
						// Bits 10..13 preserve the first literal length for the cold tail.
						h->comb[idx] = COMB_LIT | (2u << 8) | ((uint32_t)len1 << 10) | ((uint32_t)sym1 << 16) | ((uint32_t)sym2 << 24) | len;
					}
				}
			}
			if(len < FAST12_BITS) {
				memcpy(h->comb + (1u << len), h->comb, (1u << len) * sizeof(h->comb[0]));
			}
		}

	} else if(mode == 2) {
		// Zero distance entries select the canonical path for codes longer than nine bits.
		for(uint32_t i = 0; i < (1u << FAST_BITS); ++i) {
			uint32_t f = h->fast[i];
			uint32_t sym = f & 511, len = f >> 9;
			if(f && sym < 30) {
				h->dist_comb[i] = len | ((uint32_t)dist_extra[sym] << 4) | ((uint32_t)dist_base[sym] << 8);

			} else {
				h->dist_comb[i] = 0;
			}
		}
	}
	return 0;
}

// [=]===^=[ huff_decode ]=====================================================[=]
static inline int32_t huff_decode(struct bitreader *b, struct huff *h) {
	if(b->bitcnt < 16) {
		refill(b);
	}
	uint32_t f = h->fast[b->bitbuf & ((1u << FAST_BITS) - 1)];
	if(f) {
		uint32_t s = f >> 9;
		b->bitbuf >>= s;
		b->bitcnt -= s;
		return (int32_t)(f & 511);
	}
	uint32_t key = bit_reverse16((uint32_t)(b->bitbuf & 0xffff), 16);
	uint32_t s = FAST_BITS + 1;
	while((int32_t)key >= h->maxcode[s]) {
		++s;
	}
	if(s >= 16) {
		return -1;
	}
	uint32_t idx = (key >> (16 - s)) - h->firstcode[s] + h->firstsymbol[s];
	if(idx >= 288 || h->size[idx] != s) {
		return -1;
	}
	b->bitbuf >>= s;
	b->bitcnt -= s;
	return h->value[idx];
}

// [=]===^=[ huff_decode_full ]================================================[=]
// mode 1 leaves fast unbuilt, so only codes longer than 12 bits require canonical search.
static inline int32_t huff_decode_full(struct bitreader *b, struct huff *h) {
	if(b->bitcnt < 16) {
		refill(b);
	}
	uint32_t e = h->comb[b->bitbuf & ((1u << FAST12_BITS) - 1)];
	if(e & COMB_LIT) {
		// A fused pair consumes only its first code so the second remains for the next call.
		uint32_t s = ((e >> 8) & 3) == 2 ? ((e >> 10) & 0xf) : (uint8_t)e;
		b->bitbuf >>= s;
		b->bitcnt -= s;
		return (int32_t)((e >> 16) & 0xff);
	}
	if(e & COMB_EOB) {
		b->bitbuf >>= (uint8_t)e;
		b->bitcnt -= (uint8_t)e;
		return 256;
	}
	if(e) {
		b->bitbuf >>= (uint8_t)e;
		b->bitcnt -= (uint8_t)e;
		return (int32_t)(257 + ((e >> 25) & 0x1f));
	}
	uint32_t key = bit_reverse16((uint32_t)(b->bitbuf & 0xffff), 16);
	uint32_t s = 13;
	while((int32_t)key >= h->maxcode[s]) {
		++s;
	}
	if(s >= 16) {
		return -1;
	}
	uint32_t idx = (key >> (16 - s)) - h->firstcode[s] + h->firstsymbol[s];
	if(idx >= 288 || h->size[idx] != s) {
		return -1;
	}
	b->bitbuf >>= s;
	b->bitcnt -= s;
	return h->value[idx];
}

// [=]===^=[ huff_decode_nf ]==================================================[=]
static inline int32_t huff_decode_nf(struct bitreader *b, struct huff *h) {
	uint32_t f = h->fast[b->bitbuf & ((1u << FAST_BITS) - 1)];
	if(f) {
		uint32_t s = f >> 9;
		b->bitbuf >>= s;
		b->bitcnt -= s;
		return (int32_t)(f & 511);
	}
	uint32_t key = bit_reverse16((uint32_t)(b->bitbuf & 0xffff), 16);
	uint32_t s = FAST_BITS + 1;
	while((int32_t)key >= h->maxcode[s]) {
		++s;
	}
	if(s >= 16) {
		return -1;
	}
	uint32_t idx = (key >> (16 - s)) - h->firstcode[s] + h->firstsymbol[s];
	if(idx >= 288 || h->size[idx] != s) {
		return -1;
	}
	b->bitbuf >>= s;
	b->bitcnt -= s;
	return h->value[idx];
}

// [=]===^=[ copy_match ]======================================================[=]
static inline void copy_match(uint8_t *out, size_t pos, size_t distance, size_t length) {
	uint8_t *src = out + pos - distance;
	uint8_t *dst = out + pos;
	uint8_t *end = dst + length;
	if(distance >= 16) {
		// COPY_PAD permits unconditional stores, and distance at least 16 keeps every source vector behind the write frontier.
		_mm_storeu_si128((__m128i *)dst, _mm_loadu_si128((__m128i *)src));
		for(size_t i = 16; i < length; i += 16) {
			_mm_storeu_si128((__m128i *)(dst + i), _mm_loadu_si128((__m128i *)(src + i)));
		}

	} else if(distance >= 8) {
		// Distance at least eight keeps each word source behind the write frontier; overrun lands in COPY_PAD.
		do {
			memcpy(dst, src, 8);
			src += 8;
			dst += 8;
		} while(dst < end);

	} else if(distance == 1) {
		memset(dst, src[0], length);

	} else {
		// Overlapping word stores propagate periods of two through seven bytes; COPY_PAD contains temporary slop.
		do {
			memcpy(dst, src, 8);
			src += distance;
			dst += distance;
		} while(dst < end);
	}
}

// [=]===^=[ do_match ]========================================================[=]
static inline int32_t do_match(struct bitreader *b, uint32_t sym, struct huff *dist, uint8_t *out, size_t *pos, size_t outsize) {
	uint32_t li = sym - 257;
	if(li >= 29) {
		return -1;
	}
	size_t length = length_base[li] + getbits(b, length_extra[li]);
	int32_t dsym = huff_decode(b, dist);
	if(dsym < 0 || dsym >= 30) {
		return -1;
	}
	size_t distance = dist_base[dsym] + getbits(b, dist_extra[dsym]);
	if(distance > *pos || *pos + length > outsize) {
		return -1;
	}
	copy_match(out, *pos, distance, length);
	*pos += length;
	return 0;
}

// [=]===^=[ do_match_small ]==================================================[=]
__attribute__((always_inline))
static inline int32_t do_match_small(struct bitreader *b, uint32_t sym, struct huff *dist, uint8_t *out, size_t *pos, size_t outsize) {
	uint32_t li = sym - 257;
	if(li >= 29) {
		return -1;
	}
	size_t length = length_base[li] + getbits(b, length_extra[li]);
	if(b->bitcnt < 16) {
		refill(b);
	}
	uint32_t de = dist->fast[b->bitbuf & ((1u << FAST_BITS) - 1)];
	size_t distance;
	if(de) {
		uint32_t dcb = de & 15;
		uint32_t dex = (de >> 4) & 15;
		uint32_t dsym = de >> 8;
		uint32_t base = dsym < 4 ? dsym + 1 : 1 + ((2 | (dsym & 1)) << dex);
		b->bitbuf >>= dcb;
		b->bitcnt -= dcb;
		distance = base + getbits(b, dex);

	} else {
		uint32_t key = bit_reverse16((uint32_t)(b->bitbuf & 0xffff), 16);
		uint32_t s = FAST_BITS + 1;
		while((int32_t)key >= dist->maxcode[s]) {
			++s;
		}
		if(s >= 16) {
			return -1;
		}
		uint32_t idx = (key >> (16 - s)) - dist->firstcode[s] + dist->firstsymbol[s];
		if(idx >= 288 || dist->size[idx] != s) {
			return -1;
		}
		uint32_t dsym = dist->value[idx];
		if(dsym >= 30) {
			return -1;
		}
		b->bitbuf >>= s;
		b->bitcnt -= s;
		distance = dist_base[dsym] + getbits(b, dist_extra[dsym]);
	}
	if(distance > *pos || *pos + length > outsize) {
		return -1;
	}
	copy_match(out, *pos, distance, length);
	*pos += length;
	return 0;
}

// [=]===^=[ paeth ]===========================================================[=]
static uint8_t paeth(int32_t a, int32_t b, int32_t c) {
	int32_t p = a + b - c;
	int32_t pa = p > a ? p - a : a - p;
	int32_t pb = p > b ? p - b : b - p;
	int32_t pc = p > c ? p - c : c - p;
	if(pa <= pb && pa <= pc) {
		return (uint8_t)a;
	}
	if(pb <= pc) {
		return (uint8_t)b;
	}
	return (uint8_t)c;
}

// [=]===^=[ load4_u16 ]=======================================================[=]
static inline __m128i load4_u16(uint8_t *p) {
	int t;
	memcpy(&t, p, 4);
	return _mm_cvtepu8_epi16(_mm_cvtsi32_si128(t));
}

// [=]===^=[ store4 ]==========================================================[=]
static inline void store4(uint8_t *p, __m128i res) {
	int t = _mm_cvtsi128_si32(_mm_packus_epi16(res, res));
	memcpy(p, &t, 4);
}

// [=]===^=[ store3 ]==========================================================[=]
static inline void store3(uint8_t *p, __m128i res) {
	// A four-byte store would clobber the next pixel's unread input byte.
	res = _mm_packus_epi16(res, res);
	uint16_t lo = (uint16_t)_mm_extract_epi16(res, 0);
	memcpy(p, &lo, 2);
	p[2] = (uint8_t)_mm_extract_epi8(res, 2);
}

// [=]===^=[ unfilter_up ]=====================================================[=]
static void unfilter_up(uint8_t *cur, uint8_t *prev, size_t n) {
	size_t i = 0;
	for(; i + 32 <= n; i += 32) {
		__m256i c = _mm256_loadu_si256((__m256i *)(cur + i));
		__m256i p = _mm256_loadu_si256((__m256i *)(prev + i));
		_mm256_storeu_si256((__m256i *)(cur + i), _mm256_add_epi8(c, p));
	}
	for(; i < n; ++i) {
		cur[i] = (uint8_t)(cur[i] + prev[i]);
	}
}

// [=]===^=[ paeth_predict_simd ]==============================================[=]
// The u16 absolute-value form shortens the latency-bound chain; an 8-bit saturating-subtract form regressed.
static inline __m128i paeth_predict_simd(__m128i a, __m128i b, __m128i c) {
	__m128i pa = _mm_abs_epi16(_mm_sub_epi16(b, c));
	__m128i pb = _mm_abs_epi16(_mm_sub_epi16(a, c));
	__m128i pc = _mm_abs_epi16(_mm_sub_epi16(_mm_add_epi16(a, b), _mm_add_epi16(c, c)));
	__m128i not_b = _mm_cmpgt_epi16(pb, pc);
	__m128i pred = _mm_blendv_epi8(b, c, not_b);
	__m128i not_a = _mm_or_si128(_mm_cmpgt_epi16(pa, pb), _mm_cmpgt_epi16(pa, pc));
	return _mm_blendv_epi8(a, pred, not_a);
}

// [=]===^=[ paeth_simd_bpp4 ]=================================================[=]
static void paeth_simd_bpp4(uint8_t *cur, uint8_t *prev, size_t n) {
	__m128i mask = _mm_set1_epi16(0xff);
	__m128i a = _mm_setzero_si128();
	__m128i c = _mm_setzero_si128();
	for(size_t i = 0; i < n; i += 4) {
		__m128i b = load4_u16(prev + i);
		__m128i pred = paeth_predict_simd(a, b, c);
		__m128i cv = load4_u16(cur + i);
		__m128i res = _mm_and_si128(_mm_add_epi16(cv, pred), mask);
		store4(cur + i, res);
		a = res;
		c = b;
	}
}

// [=]===^=[ paeth_simd_bpp3 ]=================================================[=]
static void paeth_simd_bpp3(uint8_t *cur, uint8_t *prev, size_t n) {
	__m128i mask = _mm_set1_epi16(0xff);
	__m128i a = _mm_setzero_si128();
	__m128i c = _mm_setzero_si128();
	size_t i = 0;
	// Loads read 4 bytes, so the SIMD body stops with one pixel of slack.
	for(; i + 4 <= n; i += 3) {
		__m128i b = load4_u16(prev + i);
		__m128i pred = paeth_predict_simd(a, b, c);
		__m128i cv = load4_u16(cur + i);
		__m128i res = _mm_and_si128(_mm_add_epi16(cv, pred), mask);
		store3(cur + i, res);
		a = res;
		c = b;
	}
	for(; i < n; ++i) {
		cur[i] = (uint8_t)(cur[i] + paeth(cur[i - 3], prev[i], prev[i - 3]));
	}
}

// [=]===^=[ sub_simd_bpp4 ]===================================================[=]
static void sub_simd_bpp4(uint8_t *cur, size_t n) {
	__m128i a = _mm_setzero_si128();
	for(size_t i = 0; i < n; i += 4) {
		int ct;
		memcpy(&ct, cur + i, 4);
		__m128i res = _mm_add_epi8(_mm_cvtsi32_si128(ct), a);
		int rt = _mm_cvtsi128_si32(res);
		memcpy(cur + i, &rt, 4);
		a = res;
	}
}

// [=]===^=[ sub_simd_bpp3 ]===================================================[=]
static void sub_simd_bpp3(uint8_t *cur, size_t n) {
	__m128i a = _mm_setzero_si128();
	size_t i = 0;
	for(; i + 4 <= n; i += 3) {
		int ct;
		memcpy(&ct, cur + i, 4);
		__m128i res = _mm_add_epi8(_mm_cvtsi32_si128(ct), a);
		uint16_t lo = (uint16_t)_mm_extract_epi16(res, 0);
		memcpy(cur + i, &lo, 2);
		cur[i + 2] = (uint8_t)_mm_extract_epi8(res, 2);
		a = res;
	}
	for(; i < n; ++i) {
		cur[i] = (uint8_t)(cur[i] + cur[i - 3]);
	}
}

// [=]===^=[ avg_simd_bpp4 ]===================================================[=]
static void avg_simd_bpp4(uint8_t *cur, uint8_t *prev, size_t n) {
	// Correcting avg_epu8 by the low xor bit avoids a u16 widen and pack on the loop-carried chain.
	__m128i one = _mm_set1_epi8(1);
	__m128i a = _mm_setzero_si128();
	for(size_t i = 0; i < n; i += 4) {
		int bt, ct;
		memcpy(&bt, prev + i, 4);
		memcpy(&ct, cur + i, 4);
		__m128i b = _mm_cvtsi32_si128(bt);
		__m128i cv = _mm_cvtsi32_si128(ct);
		__m128i favg = _mm_sub_epi8(_mm_avg_epu8(a, b), _mm_and_si128(_mm_xor_si128(a, b), one));
		__m128i res = _mm_add_epi8(cv, favg);
		int rt = _mm_cvtsi128_si32(res);
		memcpy(cur + i, &rt, 4);
		a = res;
	}
}

// [=]===^=[ avg_simd_bpp3 ]===================================================[=]
static void avg_simd_bpp3(uint8_t *cur, uint8_t *prev, size_t n) {
	__m128i one = _mm_set1_epi8(1);
	__m128i a = _mm_setzero_si128();
	size_t i = 0;
	for(; i + 4 <= n; i += 3) {
		int bt, ct;
		memcpy(&bt, prev + i, 4);
		memcpy(&ct, cur + i, 4);
		__m128i b = _mm_cvtsi32_si128(bt);
		__m128i cv = _mm_cvtsi32_si128(ct);
		__m128i favg = _mm_sub_epi8(_mm_avg_epu8(a, b), _mm_and_si128(_mm_xor_si128(a, b), one));
		__m128i res = _mm_add_epi8(cv, favg);
		int rt = _mm_cvtsi128_si32(res);
		uint16_t lo = (uint16_t)rt;
		memcpy(cur + i, &lo, 2);
		cur[i + 2] = (uint8_t)(rt >> 16);
		a = res;
	}
	for(; i < n; ++i) {
		cur[i] = (uint8_t)(cur[i] + ((cur[i - 3] + prev[i]) >> 1));
	}
}

// [=]===^=[ unfilter_row ]====================================================[=]
static int32_t unfilter_row(uint8_t *cur, uint8_t *prev, uint8_t f, size_t row_bytes, uint32_t bpp) {
	switch(f) {
		case 0: {
		} break;

		case 1: {
			if(bpp == 4) {
				sub_simd_bpp4(cur, row_bytes);

			} else if(bpp == 3) {
				sub_simd_bpp3(cur, row_bytes);

			} else {
				for(size_t i = bpp; i < row_bytes; ++i) {
					cur[i] = (uint8_t)(cur[i] + cur[i - bpp]);
				}
			}
		} break;

		case 2: {
			if(prev) {
				unfilter_up(cur, prev, row_bytes);
			}
		} break;

		case 3: {
			if(prev && bpp == 4) {
				avg_simd_bpp4(cur, prev, row_bytes);

			} else if(prev && bpp == 3) {
				avg_simd_bpp3(cur, prev, row_bytes);

			} else {
				for(size_t i = 0; i < row_bytes; ++i) {
					int32_t a = i >= bpp ? cur[i - bpp] : 0;
					int32_t b = prev ? prev[i] : 0;
					cur[i] = (uint8_t)(cur[i] + ((a + b) >> 1));
				}
			}
		} break;

		case 4: {
			if(prev && bpp == 4) {
				paeth_simd_bpp4(cur, prev, row_bytes);

			} else if(prev && bpp == 3) {
				paeth_simd_bpp3(cur, prev, row_bytes);

			} else {
				for(size_t i = 0; i < row_bytes; ++i) {
					int32_t a = i >= bpp ? cur[i - bpp] : 0;
					int32_t b = prev ? prev[i] : 0;
					int32_t c = (prev && i >= bpp) ? prev[i - bpp] : 0;
					cur[i] = (uint8_t)(cur[i] + paeth(a, b, c));
				}
			}
		} break;

		default: {
			return -1;
		}
	}
	return 0;
}

// [=]===^=[ unfilter ]========================================================[=]
static int32_t unfilter(uint8_t *raw, uint32_t rows, size_t row_bytes, uint32_t bpp) {
	size_t stride = row_bytes + 1;
	uint8_t *prev = 0;
	for(uint32_t y = 0; y < rows; ++y) {
		uint8_t *cur = raw + (size_t)y * stride + 1;
		if(unfilter_row(cur, prev, raw[(size_t)y * stride], row_bytes, bpp) != 0) {
			return -1;
		}
		prev = cur;
	}
	return 0;
}

// [=]===^=[ sample_bits ]=====================================================[=]
static uint32_t sample_bits(uint8_t *row, uint32_t i, uint32_t bd) {
	uint32_t per_byte = 8 / bd;
	uint32_t byte = row[i / per_byte];
	uint32_t shift = (per_byte - 1 - (i % per_byte)) * bd;
	return (byte >> shift) & ((1u << bd) - 1);
}

// [=]===^=[ expand_row ]======================================================[=]
static void expand_row(uint8_t *src, uint8_t *dst, uint32_t width, uint32_t bd, int32_t ct, uint8_t *plte, uint8_t *trns, uint32_t trns_len, uint32_t out_ch, uint32_t *pal32) {
	if(bd == 8 && (ct == 6 || ct == 4 || (ct == 2 && out_ch == 3) || (ct == 0 && out_ch == 1))) {
		memcpy(dst, src, (size_t)width * out_ch);
		return;
	}
	switch(ct) {
		case 0: {
			uint32_t maxval = (1u << bd) - 1;
			uint32_t tgray = (trns_len >= 2) ? ((uint32_t)trns[0] << 8 | trns[1]) : 0xffffffffu;
			for(uint32_t x = 0; x < width; ++x) {
				uint32_t raw16, v8;
				if(bd == 16) {
					raw16 = (uint32_t)src[x * 2] << 8 | src[x * 2 + 1];
					v8 = src[x * 2];

				} else if(bd == 8) {
					raw16 = src[x];
					v8 = src[x];

				} else {
					uint32_t s = sample_bits(src, x, bd);
					raw16 = s;
					v8 = (uint8_t)(s * 255 / maxval);
				}
				dst[0] = (uint8_t)v8;
				if(out_ch == 2) {
					dst[1] = (raw16 == tgray) ? 0 : 0xff;
				}
				dst += out_ch;
			}
		} break;

		case 2: {
			uint32_t tr = 0xffffffffu, tg = 0, tb = 0;
			if(trns_len >= 6) {
				tr = (uint32_t)trns[0] << 8 | trns[1];
				tg = (uint32_t)trns[2] << 8 | trns[3];
				tb = (uint32_t)trns[4] << 8 | trns[5];
			}
			for(uint32_t x = 0; x < width; ++x) {
				uint32_t r16, g16, b16;
				if(bd == 16) {
					r16 = (uint32_t)src[x * 6 + 0] << 8 | src[x * 6 + 1];
					g16 = (uint32_t)src[x * 6 + 2] << 8 | src[x * 6 + 3];
					b16 = (uint32_t)src[x * 6 + 4] << 8 | src[x * 6 + 5];
					dst[0] = src[x * 6 + 0];
					dst[1] = src[x * 6 + 2];
					dst[2] = src[x * 6 + 4];

				} else {
					r16 = src[x * 3 + 0];
					g16 = src[x * 3 + 1];
					b16 = src[x * 3 + 2];
					dst[0] = src[x * 3 + 0];
					dst[1] = src[x * 3 + 1];
					dst[2] = src[x * 3 + 2];
				}
				if(out_ch == 4) {
					dst[3] = (r16 == tr && g16 == tg && b16 == tb) ? 0 : 0xff;
				}
				dst += out_ch;
			}
		} break;

		case 3: {
			if(bd == 8 && pal32 && out_ch == 4) {
				for(uint32_t x = 0; x < width; ++x) {
					memcpy(dst + (size_t)x * 4, &pal32[src[x]], 4);
				}

			} else if(bd == 8 && pal32) {
				// Overlapping four-byte stores reduce store count; the final pixel is exact to prevent row overrun.
				uint32_t x = 0;
				for(; x + 1 < width; ++x) {
					memcpy(dst, &pal32[src[x]], 4);
					dst += 3;
				}
				if(width) {
					uint32_t p = pal32[src[x]];
					dst[0] = (uint8_t)p;
					dst[1] = (uint8_t)(p >> 8);
					dst[2] = (uint8_t)(p >> 16);
				}

			} else {
				for(uint32_t x = 0; x < width; ++x) {
					uint32_t idx = (bd == 8) ? src[x] : sample_bits(src, x, bd);
					dst[0] = plte[idx * 3 + 0];
					dst[1] = plte[idx * 3 + 1];
					dst[2] = plte[idx * 3 + 2];
					if(out_ch == 4) {
						dst[3] = idx < trns_len ? trns[idx] : 0xff;
					}
					dst += out_ch;
				}
			}
		} break;

		case 4: {
			for(uint32_t x = 0; x < width; ++x) {
				if(bd == 16) {
					dst[0] = src[x * 4 + 0];
					dst[1] = src[x * 4 + 2];

				} else {
					dst[0] = src[x * 2 + 0];
					dst[1] = src[x * 2 + 1];
				}
				dst += out_ch;
			}
		} break;

		case 6: {
			for(uint32_t x = 0; x < width; ++x) {
				if(bd == 16) {
					dst[0] = src[x * 8 + 0];
					dst[1] = src[x * 8 + 2];
					dst[2] = src[x * 8 + 4];
					dst[3] = src[x * 8 + 6];

				} else {
					dst[0] = src[x * 4 + 0];
					dst[1] = src[x * 4 + 1];
					dst[2] = src[x * 4 + 2];
					dst[3] = src[x * 4 + 3];
				}
				dst += out_ch;
			}
		} break;
	}
}

// [=]===^=[ sink_consume ]====================================================[=]
// Only rows beyond window can be mutated before inflate completes; a zero window performs the final drain.
static int32_t sink_consume(struct sink *s, size_t pos, size_t window) {
	while(s->crow < s->height && (size_t)(s->crow + 1) * s->stride + window <= pos) {
		uint8_t *cur = s->raw + (size_t)s->crow * s->stride + 1;
		if(unfilter_row(cur, s->prev, s->raw[(size_t)s->crow * s->stride], s->row_bytes, s->bpp) != 0) {
			return -1;
		}
		expand_row(cur, s->pixels + (size_t)s->crow * s->width * s->out_ch, s->width, s->bd, s->ct, s->plte, s->trns, s->eff_trns, s->out_ch, s->pal32);
		s->prev = cur;
		++s->crow;
	}
	return 0;
}

// [=]===^=[ inflate_block ]===================================================[=]
// big selects the 12-bit cascade, and mode 2 delays its fourth lookup until three entries prove literal.
static int32_t inflate_block(struct bitreader *restrict b, struct huff *lit, struct huff *dist, uint8_t *restrict out, size_t outsize, size_t *outpos, uint32_t big, struct sink *sink) {
	size_t pos = *outpos;
	for(;;) {
		if(big) {
			// Indexing the unconsumed buffer pipelines three dependent literal-table loads across L1 latency.
			refill(b);
			uint32_t e1 = lit->comb[b->bitbuf & ((1u << FAST12_BITS) - 1)];
			// Chunk boundaries amortize consumer table eviction while keeping consumed rows beyond the back-reference window.
			for(;;) {
				size_t chunk_end = outsize;
				if(sink) {
					chunk_end = pos + STREAM_CHUNK;
					if(chunk_end > outsize) {
						chunk_end = outsize;
					}
				}
				while(b->in + 8 <= b->end && pos + 8 <= chunk_end) {
					if(e1 & COMB_LIT) {
						// Low-byte code lengths feed shrx directly without an extraction dependency.
						uint32_t e2 = lit->comb[(b->bitbuf >> (uint8_t)e1) & ((1u << FAST12_BITS) - 1)];
						uint32_t e3 = lit->comb[(b->bitbuf >> ((uint8_t)e1 + (uint8_t)e2)) & ((1u << FAST12_BITS) - 1)];
						uint32_t e4 = 0;
						if(__builtin_expect(big == 1, 1)) {
							e4 = lit->comb[(b->bitbuf >> ((uint8_t)e1 + (uint8_t)e2 + (uint8_t)e3)) & ((1u << FAST12_BITS) - 1)];
						}
						uint16_t v1 = (uint16_t)(e1 >> 16);
						memcpy(out + pos, &v1, 2);
						pos += (e1 >> 8) & 3;
						if(e2 & COMB_LIT) {
							uint16_t v2 = (uint16_t)(e2 >> 16);
							memcpy(out + pos, &v2, 2);
							pos += (e2 >> 8) & 3;
							if(e3 & COMB_LIT) {
								if(big > 1) {
									e4 = lit->comb[(b->bitbuf >> ((uint8_t)e1 + (uint8_t)e2 + (uint8_t)e3)) & ((1u << FAST12_BITS) - 1)];
								}
								uint16_t v3 = (uint16_t)(e3 >> 16);
								memcpy(out + pos, &v3, 2);
								pos += (e3 >> 8) & 3;
								b->bitbuf >>= (uint8_t)e1 + (uint8_t)e2 + (uint8_t)e3;
								b->bitcnt -= (uint8_t)e1 + (uint8_t)e2 + (uint8_t)e3;
								e1 = e4;
								refill(b);
								continue;
							}
							b->bitbuf >>= (uint8_t)e1 + (uint8_t)e2;
							b->bitcnt -= (uint8_t)e1 + (uint8_t)e2;
							e1 = e3;

						} else {
							b->bitbuf >>= (uint8_t)e1;
							b->bitcnt -= (uint8_t)e1;
							e1 = e2;
						}
						refill(b);
					}
					if(e1 == 0) {
						break;
					}
					b->bitbuf >>= (uint8_t)e1;
					b->bitcnt -= (uint8_t)e1;
					if(e1 & COMB_EOB) {
						*outpos = pos;
						return 0;
					}
					size_t length = ((e1 >> 16) & 0x1ff) + getbits(b, (e1 >> 8) & 7);
					uint32_t de = dist->dist_comb[b->bitbuf & ((1u << FAST_BITS) - 1)];
					size_t distance;
					if(de) {
						uint32_t dcb = de & 15;
						uint32_t dex = (de >> 4) & 15;
						distance = (de >> 8) + ((b->bitbuf >> dcb) & (((uint32_t)1 << dex) - 1));
						b->bitbuf >>= dcb + dex;
						b->bitcnt -= dcb + dex;

					} else {
						int32_t dsym = huff_decode(b, dist);
						if(dsym < 0 || dsym >= 30) {
							*outpos = pos;
							return -1;
						}
						distance = dist_base[dsym] + getbits(b, dist_extra[dsym]);
					}
					if(distance > pos || pos + length > outsize) {
						*outpos = pos;
						return -1;
					}
					// Preloading after distance consumption overlaps next-iteration setup with match-copy store latency.
					refill(b);
					e1 = lit->comb[b->bitbuf & ((1u << FAST12_BITS) - 1)];
					copy_match(out, pos, distance, length);
					pos += length;
				}
				if(b->in + 8 <= b->end && pos + 8 <= chunk_end) {
					break;
				}
				if(sink) {
					if(sink_consume(sink, pos, INFLATE_WINDOW) != 0) {
						*outpos = pos;
						return -1;
					}
					if(b->in + 8 <= b->end && chunk_end < outsize) {
						continue;
					}
				}
				break;
			}

		} else if(b->in + 8 <= b->end && pos + 3 <= outsize) {
			refill(b);
			int32_t sym = huff_decode_nf(b, lit);
			if(sym < 256) {
				out[pos++] = (uint8_t)sym;
				sym = huff_decode_nf(b, lit);
				if(sym < 256) {
					out[pos++] = (uint8_t)sym;
					sym = huff_decode_nf(b, lit);
					if(sym < 256) {
						out[pos++] = (uint8_t)sym;
						continue;
					}
				}
			}
			if(sym == 256) {
				*outpos = pos;
				return 0;
			}
			if(sym < 0) {
				*outpos = pos;
				return -1;
			}
			if(do_match_small(b, (uint32_t)sym, dist, out, &pos, outsize)) {
				*outpos = pos;
				return -1;
			}
			continue;
		}

		int32_t sym = big ? huff_decode_full(b, lit) : huff_decode(b, lit);
		if(sym < 256) {
			if(sym < 0 || pos >= outsize) {
				*outpos = pos;
				return -1;
			}
			out[pos++] = (uint8_t)sym;
			continue;
		}
		if(sym == 256) {
			*outpos = pos;
			return 0;
		}
		if(big ? do_match(b, (uint32_t)sym, dist, out, &pos, outsize) : do_match_small(b, (uint32_t)sym, dist, out, &pos, outsize)) {
			*outpos = pos;
			return -1;
		}
	}
}

// [=]===^=[ inflate_fixed_tables ]============================================[=]
static void inflate_fixed_tables(struct huff *lit, struct huff *dist, uint32_t big) {
	uint8_t lengths[288];
	uint32_t i = 0;
	for(; i < 144; ++i) {
		lengths[i] = 8;
	}
	for(; i < 256; ++i) {
		lengths[i] = 9;
	}
	for(; i < 280; ++i) {
		lengths[i] = 7;
	}
	for(; i < 288; ++i) {
		lengths[i] = 8;
	}
	huff_build(lit, lengths, 288, big ? 1 : 0);

	uint8_t dlengths[30];
	for(i = 0; i < 30; ++i) {
		dlengths[i] = 5;
	}
	huff_build(dist, dlengths, 30, big ? 2 : 3);
}

// [=]===^=[ inflate_dynamic_tables ]==========================================[=]
static int32_t inflate_dynamic_tables(struct bitreader *b, struct huff *lit, struct huff *dist, uint32_t big) {
	uint32_t hlit = getbits(b, 5) + 257;
	uint32_t hdist = getbits(b, 5) + 1;
	uint32_t hclen = getbits(b, 4) + 4;

	uint8_t cl_lengths[19] = { 0 };
	for(uint32_t i = 0; i < hclen; ++i) {
		cl_lengths[clcidx[i]] = (uint8_t)getbits(b, 3);
	}
	struct huff cl;
	if(huff_build(&cl, cl_lengths, 19, 0) != 0) {
		return -1;
	}

	uint8_t lengths[288 + 32];
	uint32_t n = 0;
	uint32_t total = hlit + hdist;
	while(n < total) {
		int32_t sym = huff_decode(b, &cl);
		if(sym < 0) {
			return -1;
		}
		if(sym < 16) {
			lengths[n++] = (uint8_t)sym;

		} else if(sym == 16) {
			if(n == 0) {
				return -1;
			}
			uint32_t rep = getbits(b, 2) + 3;
			uint8_t prev = lengths[n - 1];
			while(rep-- && n < total) {
				lengths[n++] = prev;
			}

		} else if(sym == 17) {
			uint32_t rep = getbits(b, 3) + 3;
			while(rep-- && n < total) {
				lengths[n++] = 0;
			}

		} else {
			uint32_t rep = getbits(b, 7) + 11;
			while(rep-- && n < total) {
				lengths[n++] = 0;
			}
		}
	}

	if(huff_build(lit, lengths, hlit, big ? 1 : 0) != 0 || huff_build(dist, lengths + hlit, hdist, big ? 2 : 3) != 0) {
		return -1;
	}
	return 0;
}

// [=]===^=[ inflate_stream ]====================================================[=]
static int32_t inflate_stream(uint8_t *in, size_t inlen, uint8_t *out, size_t outsize, uint32_t big, struct sink *sink) {
	if(inlen < 2) {
		return -1;
	}
	if((in[0] & 0x0f) != 8 || ((in[0] << 8 | in[1]) % 31) != 0 || (in[1] & 0x20)) {
		return -1;
	}

	struct bitreader b = { in + 2, in + inlen, 0, 0 };
	size_t outpos = 0;
	for(;;) {
		uint32_t bfinal = getbits(&b, 1);
		uint32_t btype = getbits(&b, 2);
		if(btype == 0) {
			b.bitbuf >>= b.bitcnt & 7;
			b.bitcnt -= b.bitcnt & 7;
			uint32_t len = getbits(&b, 16);
			uint32_t nlen = getbits(&b, 16);
			if((len ^ 0xffff) != nlen || outpos + len > outsize) {
				return -1;
			}
			if(len > b.bitcnt / 8 + (size_t)(b.end - b.in)) {
				return -1;
			}
			uint32_t buffered = len < b.bitcnt / 8 ? len : b.bitcnt / 8;
			for(uint32_t i = 0; i < buffered; ++i) {
				out[outpos++] = (uint8_t)getbits(&b, 8);
			}
			uint32_t remaining = len - buffered;
			if(remaining) {
				b.bitbuf = 0;
				memcpy(out + outpos, b.in, remaining);
				b.in += remaining;
				outpos += remaining;
			}

		} else if(btype == 1 || btype == 2) {
			struct huff lit, dist;
			if(btype == 1) {
				inflate_fixed_tables(&lit, &dist, big);

			} else if(inflate_dynamic_tables(&b, &lit, &dist, big)) {
				return -1;
			}
			if(inflate_block(&b, &lit, &dist, out, outsize, &outpos, big, sink)) {
				return -1;
			}

		} else {
			return -1;
		}
		if(bfinal) {
			break;
		}
	}
	return outpos == outsize ? 0 : -1;
}

// [=]===^=[ output_channels ]=================================================[=]
static uint32_t output_channels(int32_t ct, uint32_t trns_len) {
	switch(ct) {
		case 0: {
			return trns_len ? 2 : 1;
		}

		case 2: {
			return trns_len ? 4 : 3;
		}

		case 3: {
			return trns_len ? 4 : 3;
		}

		case 4: {
			return 2;
		}

		case 6: {
			return 4;
		}
	}
	return 0;
}

// [=]===^=[ raw_channels ]====================================================[=]
static uint32_t raw_channels(int32_t ct) {
	switch(ct) {
		case 0: {
			return 1;
		}

		case 2: {
			return 3;
		}

		case 3: {
			return 1;
		}

		case 4: {
			return 2;
		}

		case 6: {
			return 4;
		}
	}
	return 0;
}

// [=]===^=[ pd_decode ]======================================================[=]
int pd_decode(uint8_t *data, size_t len, struct pd_image *out) {
	static uint8_t sig[8] = { 0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a };
	if(!crc_ready) {
		crc_build();
	}
	if(len < 8 || memcmp(data, sig, 8) != 0) {
		return 1;
	}

	uint32_t width = 0, height = 0;
	uint32_t bd = 0;
	int32_t ct = -1;
	uint32_t interlace = 0;
	uint8_t plte[256 * 3];
	uint8_t trns[256];
	uint32_t trns_len = 0;
	uint32_t have_ihdr = 0;

	uint8_t *idat = 0;
	size_t idat_len = 0, idat_cap = 0;

	size_t pos = 8;
	uint32_t saw_iend = 0;
	while(pos + 8 <= len) {
		uint32_t clen = (uint32_t)data[pos] << 24 | (uint32_t)data[pos + 1] << 16 | (uint32_t)data[pos + 2] << 8 | data[pos + 3];
		uint8_t *type = data + pos + 4;
		if(pos + 12 + clen > len) {
			break;
		}
		uint8_t *cdata = data + pos + 8;
		// Ancillary CRC failures are tolerated to match image-png's skip_ancillary_crc_failures behavior.
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
		uint32_t raw_owned = raw_size < RAW_REUSE_MIN;
		uint8_t *raw = raw_owned ? malloc(raw_size + COPY_PAD) : raw_grow(raw_size);
		struct sink sink = { raw, pixels, 0, plte, trns, pal32_ptr, stride, row_bytes, 0, height, width, bpp, bd, out_ch, eff_trns, ct };
		struct sink *sp = raw_size > STREAM_MIN ? &sink : 0;
		uint32_t big = raw_size >= BIG_MIN ? (raw_size / 5 < idat_len / 2 ? 2 : 1) : 0;
		if(!raw || inflate_stream(idat, idat_len, raw, raw_size, big, sp) != 0 || sink_consume(&sink, raw_size, 0) != 0) {
			if(raw_owned) {
				free(raw);
			}
			free(idat);
			free(pixels);
			return 1;
		}
		if(raw_owned) {
			free(raw);
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
		uint32_t raw_owned = raw_size < RAW_REUSE_MIN;
		uint8_t *raw = raw_owned ? malloc(raw_size + COPY_PAD) : raw_grow(raw_size);
		uint8_t *temp = malloc((size_t)maxw * out_ch);
		uint32_t big = raw_size >= BIG_MIN ? (raw_size / 5 < idat_len / 2 ? 2 : 1) : 0;
		if(!raw || !temp || inflate_stream(idat, idat_len, raw, raw_size, big, 0) != 0) {
			if(raw_owned) {
				free(raw);
			}
			free(temp);
			free(idat);
			free(pixels);
			return 1;
		}
		uint32_t ok = 1;
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
		if(raw_owned) {
			free(raw);
		}
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

// [=]===^=[ pd_free ]========================================================[=]
void pd_free(struct pd_image *img) {
	free(img->pixels);
	img->pixels = 0;
}
