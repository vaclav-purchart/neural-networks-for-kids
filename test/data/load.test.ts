import { describe, expect, it } from "vitest"
import { DATASETS } from "../../src/data/datasets"
import { makeSplit, pixelStats } from "../../src/data/load"

describe("makeSplit", () => {
	it("applies the EMNIST transpose and label offset, and respects limit", () => {
		const images = { shape: [3, 2, 2], data: Uint8Array.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]) }
		const labels = { shape: [3], data: Uint8Array.from([1, 2, 26]) }
		const split = makeSplit(DATASETS["emnist-letters"], images, labels, 2, 2, 2)
		expect(split.count).toBe(2)
		expect([...split.images]).toEqual([1, 3, 2, 4, 5, 7, 6, 8])
		expect([...split.labels]).toEqual([0, 1])
		expect([...images.data.subarray(0, 4)]).toEqual([1, 2, 3, 4]) // source untouched
	})

	it("rejects mismatched image and label counts", () => {
		const images = { shape: [2, 1, 1], data: Uint8Array.from([0, 0]) }
		const labels = { shape: [3], data: Uint8Array.from([0, 0, 0]) }
		expect(() => makeSplit(DATASETS.mnist, images, labels, 1, 1)).toThrow(/2 images but 3 labels/)
	})
})

describe("pixelStats", () => {
	it("computes mean and std of 0..1 pixel values", () => {
		const { mean, std } = pixelStats(Uint8Array.from([0, 255]))
		expect(mean).toBeCloseTo(0.5)
		expect(std).toBeCloseTo(0.5)
	})
})
