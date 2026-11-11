import { describe, expect, it } from "vitest"
import { FAST_MS, Player, playbackUnit, rowBrightness, STEPS, speedToDelay } from "../../src/web/player"

describe("Player", () => {
	it("has the seven stages of one prediction", () => {
		expect(STEPS).toHaveLength(7)
		expect(STEPS[0]).toMatch(/Text/)
		expect(STEPS[6]).toMatch(/Pick/)
	})

	it("steps forward through ① … ⑦, then asks to append the picked letter and starts over", () => {
		const p = new Player()
		const results = Array.from({ length: 7 }, () => p.forward())
		expect(results.slice(0, 6)).toEqual([null, null, null, null, null, null])
		expect(p.step).toBe(0)
		expect(results[6]).toBe("append")
	})

	it("goes back one stage but not before ①", () => {
		const p = new Player()
		p.jumpTo(3)
		p.back()
		expect(p.step).toBe(2)
		p.jumpTo(0)
		p.back()
		expect(p.step).toBe(0)
	})

	it("clamps jumps and resets to ①", () => {
		const p = new Player()
		p.jumpTo(9)
		expect(p.step).toBe(6)
		p.jumpTo(-2)
		expect(p.step).toBe(0)
		p.jumpTo(4)
		p.reset()
		expect(p.step).toBe(0)
	})
})

describe("speedToDelay", () => {
	it("goes from 3 s per step (slowest) to 20 ms (fastest) on a log scale", () => {
		expect(speedToDelay(0)).toBeCloseTo(3000)
		expect(speedToDelay(1)).toBeCloseTo(20)
		expect(speedToDelay(0.5)).toBeCloseTo(Math.sqrt(3000 * 20))
	})
	it("clamps out-of-range input", () => {
		expect(speedToDelay(-1)).toBeCloseTo(3000)
		expect(speedToDelay(2)).toBeCloseTo(20)
	})
})

describe("playbackUnit", () => {
	it("plays step by step when slow enough to follow, whole letters when fast", () => {
		expect(playbackUnit(speedToDelay(0))).toBe("step")
		expect(playbackUnit(FAST_MS)).toBe("step")
		expect(playbackUnit(FAST_MS - 1)).toBe("letter")
		expect(playbackUnit(speedToDelay(1))).toBe("letter")
	})
})

describe("rowBrightness", () => {
	it("lights every row fully when not stepping", () => {
		for (let row = 0; row < STEPS.length; row++) expect(rowBrightness(row, 2, false)).toBe(1)
	})
	it("focuses the active row while stepping, keeping the others visible", () => {
		expect(rowBrightness(2, 2, true)).toBe(1)
		expect(rowBrightness(1, 2, true)).toBe(0.5) // finished
		expect(rowBrightness(5, 2, true)).toBe(0.2) // upcoming: dim but readable
	})
})
