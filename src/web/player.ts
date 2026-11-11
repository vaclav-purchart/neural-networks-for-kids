/** The stages of one prediction, in the order the player walks through them. */
export const STEPS = [
	"① Text",
	"② Embedding",
	"③ Attention · block 1",
	"④ MLP · block 1",
	"⑤ Block 2",
	"⑥ Output",
	"⑦ Pick a letter",
] as const

const LAST = STEPS.length - 1
const SLOWEST_MS = 3000
const FASTEST_MS = 20

/** Which stage of the current prediction is shown. Timing and playing are the caller's job. */
export class Player {
	step = 0

	/** Next stage; leaving ⑦ returns "append": add the picked letter, the next prediction starts at ①. */
	forward(): "append" | null {
		if (this.step < LAST) {
			this.step++
			return null
		}
		this.step = 0
		return "append"
	}

	back(): void {
		this.step = Math.max(0, this.step - 1)
	}

	jumpTo(step: number): void {
		this.step = Math.min(LAST, Math.max(0, Math.round(step)))
	}

	reset(): void {
		this.step = 0
	}
}

/** Speed slider (0 = slowest … 1 = fastest) → milliseconds per step, on a log scale. */
export function speedToDelay(speed: number): number {
	const v = Math.min(1, Math.max(0, speed))
	return SLOWEST_MS * (FASTEST_MS / SLOWEST_MS) ** v
}

/** Faster than this (ms per tick), single steps can't be followed: playback writes whole letters. */
export const FAST_MS = 150

/** What one playback tick does at this delay: advance one step, or write one whole letter. */
export function playbackUnit(delayMs: number): "step" | "letter" {
	return delayMs < FAST_MS ? "letter" : "step"
}

/**
 * Brightness of a row of the network: all rows full when not stepping; while stepping the active row is
 * full, finished rows half, upcoming rows dim but still readable.
 */
export function rowBrightness(row: number, step: number, stepping: boolean): number {
	if (!stepping || row === step) return 1
	return row < step ? 0.5 : 0.2
}
