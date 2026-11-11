/** IEEE-754 half precision (float16) conversion and compact base64 packing for weights. */

const f32 = new Float32Array(1)
const u32 = new Uint32Array(f32.buffer)

/** float → float16 bits (round to nearest). */
export function toHalf(value: number): number {
	f32[0] = value
	const x = u32[0]
	const sign = (x >>> 16) & 0x8000
	const exp = (x >>> 23) & 0xff
	let mant = x & 0x7fffff
	if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0) // Infinity / NaN
	let e = exp - 127 + 15
	if (e >= 0x1f) return sign | 0x7c00 // too large → Infinity
	if (e <= 0) {
		if (e < -10) return sign // too small → ±0
		mant = (mant | 0x800000) >> (1 - e) // subnormal
		if (mant & 0x1000) mant += 0x2000
		return sign | (mant >> 13)
	}
	if (mant & 0x1000) {
		mant += 0x2000
		if (mant & 0x800000) {
			mant = 0
			e += 1
			if (e >= 0x1f) return sign | 0x7c00
		}
	}
	return sign | (e << 10) | (mant >> 13)
}

/** float16 bits → number. */
export function fromHalf(h: number): number {
	const sign = h & 0x8000 ? -1 : 1
	const e = (h >> 10) & 0x1f
	const m = h & 0x3ff
	if (e === 0) return sign * m * 2 ** -24
	if (e === 0x1f) return m ? NaN : sign * Infinity
	return sign * (1 + m / 1024) * 2 ** (e - 15)
}

function bytesToBase64(bytes: Uint8Array): string {
	let s = ""
	const CHUNK = 0x8000
	for (let i = 0; i < bytes.length; i += CHUNK) s += String.fromCharCode(...bytes.subarray(i, i + CHUNK))
	return btoa(s)
}

function base64ToBytes(text: string): Uint8Array {
	const s = atob(text)
	const out = new Uint8Array(s.length)
	for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i)
	return out
}

/** Pack numbers as little-endian float16, base64-encoded. */
export function encodeFloat16(values: ArrayLike<number>): string {
	const bytes = new Uint8Array(values.length * 2)
	for (let i = 0; i < values.length; i++) {
		const h = toHalf(values[i])
		bytes[2 * i] = h & 0xff
		bytes[2 * i + 1] = h >> 8
	}
	return bytesToBase64(bytes)
}

export function decodeFloat16(text: string): Float32Array {
	const bytes = base64ToBytes(text)
	if (bytes.length % 2) throw new Error("float16 data has an odd byte length")
	const out = new Float32Array(bytes.length / 2)
	for (let i = 0; i < out.length; i++) out[i] = fromHalf(bytes[2 * i] | (bytes[2 * i + 1] << 8))
	return out
}
