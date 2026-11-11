import { describe, expect, it } from "vitest"
import { topK } from "../../src/web/chart"

describe("topK", () => {
	it("returns the k most likely indices, highest first", () => {
		expect(topK([0.1, 0.5, 0.05, 0.3, 0.05], 3)).toEqual([1, 3, 0])
	})
	it("returns all indices when k exceeds the length", () => {
		expect(topK([0.2, 0.8], 5)).toEqual([1, 0])
	})
})
