// Copyright (c) 2026 Peter Fors
// SPDX-License-Identifier: MIT
#include <stdint.h>
#include <stddef.h>

// Decoded image. `pixels` is 8-bit, width*height*channels bytes, EXPAND|STRIP_16
// semantics (palette to RGB, sub-8-bit gray to 8-bit, tRNS to alpha, 16-bit
// truncated to high byte). channels: 1=gray 2=gray+a 3=rgb 4=rgba.
struct pd_image {
	uint8_t *pixels;
	uint32_t width;
	uint32_t height;
	uint8_t channels;
};

// Decodes a complete PNG from memory. Returns 0 on success; on success the
// caller owns `out->pixels` and frees it with pd_free.
int pd_decode(uint8_t *data, size_t len, struct pd_image *out);

void pd_free(struct pd_image *img);
