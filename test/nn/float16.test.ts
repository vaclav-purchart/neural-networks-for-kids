import { describe, expect, it } from "vitest"
import { decodeFloat16, encodeFloat16, fromHalf, toHalf } from "../../src/nn/float16"
import { mulberry32 } from "../../src/nn/rng"

describe("float16", () => {
	it("encodes known values exactly", () => {
		expect(toHalf(1)).toBe(0x3c00)
		expect(toHalf(-2)).toBe(0xc000)
		expect(toHalf(0)).toBe(0)
		expect(toHalf(65504)).toBe(0x7bff)
		expect(toHalf(1e6)).toBe(0x7c00) // overflow → +Infinity
		expect(fromHalf(0x3c00)).toBe(1)
		expect(fromHalf(0x7c00)).toBe(Infinity)
	})

	it("round-trips arrays within half precision", () => {
		const rand = mulberry32(3)
		const values = Float32Array.from({ length: 1000 }, () => (rand() - 0.5) * 8)
		const decoded = decodeFloat16(encodeFloat16(values))
		expect(decoded.length).toBe(values.length)
		for (let i = 0; i < values.length; i++) {
			expect(Math.abs(decoded[i] - values[i])).toBeLessThanOrEqual(Math.abs(values[i]) * 1e-3 + 1e-7)
		}
	})

	it("handles subnormal values", () => {
		expect(Math.abs(fromHalf(toHalf(1e-6)) - 1e-6)).toBeLessThan(6e-8)
	})

	it("rejects odd byte lengths", () => {
		expect(() => decodeFloat16(btoa("abc"))).toThrow(/odd/)
	})
})
