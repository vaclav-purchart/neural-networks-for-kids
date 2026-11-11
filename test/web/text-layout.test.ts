import { describe, expect, it } from "vitest"
import { arcPoints, letterX } from "../../src/web/text-scene"
import { VISIBLE_LETTERS } from "../../src/web/text-view-model"

describe("transformer view layout", () => {
	it("keeps the newest letter in the rightmost slot and the slots centred", () => {
		const right = letterX(VISIBLE_LETTERS - 1, VISIBLE_LETTERS)
		expect(letterX(0, VISIBLE_LETTERS)).toBeCloseTo(-right)
		expect(letterX(2, 3)).toBeCloseTo(right) // with only 3 letters the newest is still rightmost
		expect(letterX(-1, 3)).toBeLessThan(letterX(0, 3)) // the "earlier text" slot is left of the first letter
	})

	it("arcs start and end at the given points and bulge towards the camera", () => {
		const p = arcPoints({ x: 0, y: 1, z: 0 }, { x: 4, y: 3, z: 0 }, 1, 10)
		expect(p).toHaveLength(11 * 3)
		expect(p.slice(0, 3)).toEqual([0, 1, 0])
		expect(p.slice(-3)).toEqual([4, 3, 0])
		expect(p[5 * 3 + 2]).toBeGreaterThan(0.9)
	})
})
