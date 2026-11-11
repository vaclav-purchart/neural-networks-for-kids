import type { Rand } from "../rng"
import type { Transformer } from "./model"
import type { Vocab } from "./tokenizer"

/** Draw a token from logits[offset … offset+V); temperature ≈ 0 means "always the most likely". */
export function sampleFromLogits(
	logits: ArrayLike<number>,
	offset: number,
	V: number,
	temperature: number,
	rand: Rand,
): number {
	let best = 0
	for (let i = 1; i < V; i++) if (logits[offset + i] > logits[offset + best]) best = i
	if (temperature <= 1e-6) return best
	const max = logits[offset + best]
	const weights = new Float64Array(V)
	let sum = 0
	for (let i = 0; i < V; i++) {
		weights[i] = Math.exp((logits[offset + i] - max) / temperature)
		sum += weights[i]
	}
	let r = rand() * sum
	for (let i = 0; i < V; i++) {
		r -= weights[i]
		if (r < 0) return i
	}
	return V - 1
}

/** Where generation starts when there is no prompt: a new line if the model knows one. */
export function startId(vocab: Vocab): number {
	return vocab.index.get("\n") ?? 0
}

/** Generate `count` new token ids after `prompt` (only the last `context` ids are fed each step). */
export function generate(
	model: Transformer,
	prompt: readonly number[],
	count: number,
	temperature: number,
	rand: Rand,
	fallback: number,
): number[] {
	const { context, vocab } = model.config
	const ids = prompt.length ? [...prompt] : [fallback]
	const out: number[] = []
	for (let i = 0; i < count; i++) {
		const window = ids.slice(-context)
		const fwd = model.forward(window, 1, window.length)
		const next = sampleFromLogits(fwd.logits, (window.length - 1) * vocab, vocab, temperature, rand)
		ids.push(next)
		out.push(next)
	}
	return out
}
