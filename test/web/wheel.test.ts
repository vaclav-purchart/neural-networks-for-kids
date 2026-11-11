import { describe, expect, it } from "vitest"
import { wheelAction, zoomFactor } from "../../src/web/wheel"

const ev = (deltaX: number, deltaY: number, ctrlKey = false, deltaMode = 0) => ({ deltaX, deltaY, ctrlKey, deltaMode })

describe("wheelAction", () => {
	it("pans for a two-finger touchpad swipe (small or fractional deltas, or sideways)", () => {
		expect(wheelAction(ev(0, 4))).toBe("pan")
		expect(wheelAction(ev(0, 37.5))).toBe("pan")
		expect(wheelAction(ev(12, 0))).toBe("pan")
		expect(wheelAction(ev(-3, 120))).toBe("pan") // any sideways movement means a touchpad
	})
	it("zooms for a pinch (the browser sets ctrlKey)", () => {
		expect(wheelAction(ev(0, 3.2, true))).toBe("zoom")
	})
	it("zooms for a mouse wheel (big whole-number steps, or line mode)", () => {
		expect(wheelAction(ev(0, 100))).toBe("zoom")
		expect(wheelAction(ev(0, -120))).toBe("zoom")
		expect(wheelAction(ev(0, 3, false, 1))).toBe("zoom")
	})
})

describe("zoomFactor", () => {
	it("moves closer for negative deltas and away for positive ones", () => {
		expect(zoomFactor(-100, false)).toBeLessThan(1)
		expect(zoomFactor(100, false)).toBeGreaterThan(1)
		expect(zoomFactor(0, true)).toBe(1)
	})
	it("treats a small pinch like a bigger wheel step", () => {
		expect(zoomFactor(10, true)).toBeCloseTo(zoomFactor(100, false))
	})
})
