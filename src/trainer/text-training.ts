import type { Transformer } from "../nn/transformer/model"
import type { Vocab } from "../nn/transformer/tokenizer"

const EARLY_STEPS = [0, 10, 25, 50, 100, 200, 400]
const LATER_SNAPSHOTS = 12
/** Evenly spaced snapshots closer together than this many steps aren't worth taking. */
const MIN_LATER_GAP = 10

/** Steps after which a snapshot is taken: dense early (where text changes fastest), then evenly spaced. */
export function snapshotSteps(total: number): number[] {
	const points = new Set(EARLY_STEPS.filter((s) => s < total))
	if (total / LATER_SNAPSHOTS >= MIN_LATER_GAP) {
		for (let i = 1; i <= LATER_SNAPSHOTS; i++) points.add(Math.round((total * i) / LATER_SNAPSHOTS))
	}
	points.add(total)
	return [...points].sort((a, b) => a - b)
}

/** Linear decay from base to 10% of base. */
export function lrAt(step: number, total: number, base: number): number {
	return base * (1 - 0.9 * Math.min(1, step / total))
}

/** x[n] = ids[start … start+T), y[n] = the same window shifted by one. */
export function fillBatch(ids: Uint16Array, starts: ArrayLike<number>, T: number, x: Int32Array, y: Int32Array): void {
	for (let n = 0; n < starts.length; n++) {
		const s = starts[n]
		for (let t = 0; t < T; t++) {
			x[n * T + t] = ids[s + t]
			y[n * T + t] = ids[s + t + 1]
		}
	}
}

/** Up to `count` window starts spread evenly over a text of `length` (each needs T + 1 characters). */
export function evalWindows(length: number, T: number, count: number): number[] {
	const last = length - T - 1
	if (last < 0) return []
	const n = Math.min(count, last + 1)
	if (n === 1) return [0]
	return Array.from({ length: n }, (_, i) => Math.round((i * last) / (n - 1)))
}

/** Mean loss and next-character accuracy over every position of the given windows. */
export function evaluateText(
	model: Transformer,
	ids: Uint16Array,
	starts: readonly number[],
	T: number,
): { loss: number; accuracy: number; samples: number } {
	const V = model.config.vocab
	const B = 32
	let lossSum = 0
	let correct = 0
	let samples = 0
	for (let i = 0; i < starts.length; i += B) {
		const chunk = starts.slice(i, i + B)
		const x = new Int32Array(chunk.length * T)
		const y = new Int32Array(chunk.length * T)
		fillBatch(ids, chunk, T, x, y)
		const fwd = model.forward(x, chunk.length, T)
		lossSum += model.loss(fwd, y) * chunk.length * T
		for (let r = 0; r < chunk.length * T; r++) {
			let best = 0
			for (let c = 1; c < V; c++) if (fwd.probs[r * V + c] > fwd.probs[r * V + best]) best = c
			if (best === y[r]) correct++
		}
		samples += chunk.length * T
	}
	return { loss: samples ? lossSum / samples : 0, accuracy: samples ? correct / samples : 0, samples }
}

/** First line of the test text (with its newline), at most maxLength characters and at least one. */
export function promptIds(test: Uint16Array, vocab: Vocab, maxLength = 32): number[] {
	const newline = vocab.index.get("\n")
	const out: number[] = []
	for (const id of test) {
		out.push(id)
		if (id === newline || out.length >= maxLength) break
	}
	return out
}
