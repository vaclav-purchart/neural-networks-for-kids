/** The parts of a WheelEvent that tell a touchpad swipe, a pinch and a mouse wheel apart. */
export interface WheelLike {
	deltaX: number
	deltaY: number
	deltaMode: number
	ctrlKey: boolean
}

/**
 * Browsers report all three as "wheel" events: a pinch has ctrlKey set; a mouse wheel moves in big
 * whole-number steps (or whole lines) with no sideways part; anything else is a two-finger swipe.
 */
export function wheelAction(e: WheelLike): "pan" | "zoom" {
	if (e.ctrlKey) return "zoom"
	if (e.deltaMode !== 0) return "zoom"
	if (e.deltaX !== 0) return "pan"
	return Number.isInteger(e.deltaY) && Math.abs(e.deltaY) >= 50 ? "zoom" : "pan"
}

/** Camera distance multiplier for a zoom wheel event (< 1 = closer). Pinch deltas are ~10× smaller. */
export function zoomFactor(deltaY: number, pinch: boolean): number {
	return Math.exp(deltaY * (pinch ? 0.01 : 0.001))
}
