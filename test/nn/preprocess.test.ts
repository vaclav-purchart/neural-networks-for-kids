import { describe, expect, it } from "vitest"
import { centerByBox, centerByMass, normalize, normalizeSize, prepareDrawing } from "../../src/nn/preprocess"

describe("centerByMass", () => {
	it("returns zeros (no NaN) for a blank image", () => {
		const out = centerByMass(new Float32Array(28 * 28), 28, 28)
		expect(out.every((v) => v === 0)).toBe(true)
	})

	it("moves an off-centre stroke to the middle", () => {
		const img = new Float32Array(28 * 28)
		img[0] = 1 // top-left corner
		const out = centerByMass(img, 28, 28)
		expect(out[14 * 28 + 14]).toBe(1)
		expect(out.reduce((a, b) => a + b, 0)).toBe(1)
	})

	it("leaves a centred image unchanged", () => {
		const img = new Float32Array(9)
		img[4] = 1
		expect([...centerByMass(img, 3, 3)]).toEqual([...img])
	})
})

describe("normalize", () => {
	it("applies (x - mean) / std", () => {
		expect([...normalize([0, 1], 0.5, 0.25)]).toEqual([-2, 2])
	})
})

/** Height and width of the area with ink (> 0.05). */
function inkBox(img: Float32Array, rows: number, cols: number): { height: number; width: number } {
	let top = rows,
		bottom = -1,
		left = cols,
		right = -1
	for (let r = 0; r < rows; r++) {
		for (let c = 0; c < cols; c++) {
			if (img[r * cols + c] > 0.05) {
				top = Math.min(top, r)
				bottom = Math.max(bottom, r)
				left = Math.min(left, c)
				right = Math.max(right, c)
			}
		}
	}
	return { height: bottom - top + 1, width: right - left + 1 }
}

function fill(rows: [number, number], cols: [number, number]): Float32Array {
	const img = new Float32Array(28 * 28)
	for (let r = rows[0]; r <= rows[1]; r++) for (let c = cols[0]; c <= cols[1]; c++) img[r * 28 + c] = 1
	return img
}

describe("normalizeSize", () => {
	it("passes a blank image through (no NaN)", () => {
		const out = normalizeSize(new Float32Array(784), 28, 28, 20)
		expect(out.every((v) => v === 0)).toBe(true)
	})

	it("enlarges a tiny drawing to the box size", () => {
		const out = normalizeSize(fill([2, 5], [2, 5]), 28, 28, 20)
		const box = inkBox(out, 28, 28)
		expect(box.height).toBeGreaterThanOrEqual(19)
		expect(box.height).toBeLessThanOrEqual(21)
		expect(out.every((v) => Number.isFinite(v) && v >= 0 && v <= 1)).toBe(true)
	})

	it("shrinks a drawing that fills the whole field", () => {
		const box = inkBox(normalizeSize(fill([0, 27], [0, 27]), 28, 28, 20), 28, 28)
		expect(box.height).toBeGreaterThanOrEqual(19)
		expect(box.height).toBeLessThanOrEqual(21)
	})

	it("keeps the aspect ratio (a thin '1' stays thin)", () => {
		const box = inkBox(normalizeSize(fill([9, 18], [13, 14]), 28, 28, 20), 28, 28)
		expect(box.height).toBeGreaterThanOrEqual(19)
		expect(box.width).toBeLessThanOrEqual(5)
	})

	it("leaves a drawing whose ink already fits the box unchanged", () => {
		const img = fill([4, 23], [10, 17]) // 20 tall, already centred
		const before = centerByMass(img, 28, 28)
		const after = centerByMass(normalizeSize(img, 28, 28, 20), 28, 28)
		expect(Math.max(...after.map((v, i) => Math.abs(v - before[i])))).toBeLessThan(1e-6)
	})

	it("uses a larger box for letters", () => {
		expect(inkBox(normalizeSize(fill([2, 5], [2, 5]), 28, 28, 24), 28, 28).height).toBeGreaterThanOrEqual(23)
	})
})

describe("normalizeSize with almost no ink", () => {
	it("does not blow a single tap up into a big block", () => {
		const dot = new Float32Array(784)
		dot[10 * 28 + 10] = 1
		expect([...normalizeSize(dot, 28, 28, 20)]).toEqual([...dot])
		const twoByTwo = fill([5, 6], [5, 6])
		expect([...normalizeSize(twoByTwo, 28, 28, 20)]).toEqual([...twoByTwo])
	})
})

/** Top-left corner of the ink box. */
function inkTopLeft(img: Float32Array): [number, number] {
	let top = 28
	let left = 28
	img.forEach((v, i) => {
		if (v > 0.05) {
			top = Math.min(top, Math.floor(i / 28))
			left = Math.min(left, i % 28)
		}
	})
	return [top, left]
}

describe("centerByBox", () => {
	it("returns zeros (no NaN) for a blank image", () => {
		expect(centerByBox(new Float32Array(784), 28, 28).every((v) => v === 0)).toBe(true)
	})

	it("moves the ink box to the middle without scaling it", () => {
		const out = centerByBox(fill([0, 2], [0, 4]), 28, 28) // 3 tall, 5 wide, in the corner
		expect(inkTopLeft(out)).toEqual([12, 11])
		expect(inkBox(out, 28, 28)).toEqual({ height: 3, width: 5 })
	})

	it("ignores where the mass is: a lopsided shape stays box-centred", () => {
		const img = fill([10, 17], [10, 17]) // centred 8×8 box…
		for (let r = 10; r <= 17; r++) for (let c = 10; c <= 13; c++) img[r * 28 + c] = 0.2 // …with a light left half
		expect([...centerByBox(img, 28, 28)]).toEqual([...img])
	})
})

describe("prepareDrawing", () => {
	const corner = fill([0, 5], [0, 3])

	it("digits (mass): scales, then centres by centre of mass", () => {
		const pre = { fitBox: 20, center: "mass" as const }
		const expected = centerByMass(normalizeSize(corner, 28, 28, 20), 28, 28)
		expect([...prepareDrawing(corner, 28, 28, pre, true)]).toEqual([...expected])
		expect([...prepareDrawing(corner, 28, 28, pre, false)]).toEqual([...centerByMass(corner, 28, 28)])
	})

	it("letters (box): scales and box-centres, or only box-centres when scaling is off", () => {
		const pre = { fitBox: 24, center: "box" as const }
		expect(inkBox(prepareDrawing(corner, 28, 28, pre, true), 28, 28).height).toBeGreaterThanOrEqual(23)
		expect([...prepareDrawing(corner, 28, 28, pre, false)]).toEqual([...centerByBox(corner, 28, 28)])
	})
})
