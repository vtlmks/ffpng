// Copyright (c) 2026 Peter Fors
// SPDX-License-Identifier: MIT
//
// C ABI shim over the image-png decoder, built with `--features=unstable`
// (portable_simd) and `target-cpu=native`, so the C harness can time and
// validate it in the same process as every other decoder.
//
// Output convention matches the rest of the harness: native EXPAND | STRIP_16
// (the `image` crate default), i.e. 8-bit samples, channel count left native.

use std::io::Cursor;
use std::mem::ManuallyDrop;
use std::slice;

#[repr(C)]
pub struct ShimResult {
	pixels: *mut u8,
	len: usize,
	width: u32,
	height: u32,
	channels: u8,
}

#[no_mangle]
pub extern "C" fn imagepng_decode(data: *const u8, len: usize, out: *mut ShimResult) -> i32 {
	let input = unsafe { slice::from_raw_parts(data, len) };
	let mut decoder = png::Decoder::new(Cursor::new(input));
	decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);

	let mut reader = match decoder.read_info() {
		Ok(r) => r,
		Err(_) => return 1,
	};
	let size = match reader.output_buffer_size() {
		Some(s) => s,
		None => return 1,
	};
	let mut buf = vec![0u8; size];
	let info = match reader.next_frame(&mut buf) {
		Ok(i) => i,
		Err(_) => return 1,
	};

	// vec![0u8; size] guarantees len == capacity == size, so the C side can
	// reconstruct the Vec from (ptr, size, size) to free it.
	let channels = info.color_type.samples() as u8;
	let width = info.width;
	let height = info.height;
	let mut held = ManuallyDrop::new(buf);
	unsafe {
		(*out).pixels = held.as_mut_ptr();
		(*out).len = held.len();
		(*out).width = width;
		(*out).height = height;
		(*out).channels = channels;
	}
	0
}

#[no_mangle]
pub extern "C" fn imagepng_free(pixels: *mut u8, len: usize) {
	if !pixels.is_null() {
		unsafe {
			drop(Vec::from_raw_parts(pixels, len, len));
		}
	}
}
