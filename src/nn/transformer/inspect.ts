import type { TransformerConfig, TransformerForward } from "./model"

export interface AttendedChar {
	position: number
	weight: number
}

export interface HeadAttention {
	block: number
	head: number
	/** Strongest attended positions, largest weight first. */
	top: AttendedChar[]
}

export interface PositionDetails {
	position: number
	heads: HeadAttention[]
	/** Most likely next tokens after this position, most likely first. */
	predictions: { id: number; p: number }[]
}

/** What the model does at one position of a single sequence (batch 1). */
export function inspectPosition(
	fwd: TransformerForward,
	config: TransformerConfig,
	position: number,
	topK = 3,
): PositionDetails {
	const { T } = fwd
	const H = config.heads
	const heads: HeadAttention[] = []
	fwd.attention.forEach((att, block) => {
		for (let head = 0; head < H; head++) {
			const row = (head * T + position) * T
			const top = Array.from({ length: position + 1 }, (_, s) => ({ position: s, weight: att[row + s] }))
				.sort((a, b) => b.weight - a.weight)
				.slice(0, topK)
			heads.push({ block, head, top })
		}
	})
	const V = config.vocab
	const predictions = Array.from({ length: V }, (_, id) => ({ id, p: fwd.probs[position * V + id] }))
		.sort((a, b) => b.p - a.p)
		.slice(0, topK)
	return { position, heads, predictions }
}
