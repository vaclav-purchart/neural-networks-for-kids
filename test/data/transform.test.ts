import { describe, expect, it } from "vitest"
import { remapLabels, transposeImages } from "../../src/data/transform"

describe("transforms", () => {
	it("transposes each image independently", () => {
		const imgs = Uint8Array.from([1, 2, 3, 4, 5, 6, 7, 8]) // two 2×2 images
		transposeImages(imgs, 2, 2)
		expect([...imgs]).toEqual([1, 3, 2, 4, 5, 7, 6, 8])
	})

	it("shifts EMNIST letter labels 1..26 to 0..25 and rejects out-of-range labels", () => {
		expect([...remapLabels(Uint8Array.from([1, 26]), -1, 26)]).toEqual([0, 25])
		expect(() => remapLabels(Uint8Array.from([0]), -1, 26)).toThrow(/outside 0\.\.25/)
	})
})
