import { paramShapes, type Transformer, type TransformerConfig, type TransformerForward } from "../nn/transformer/model"

/** How many of the latest characters the 3D view draws. */
export const VISIBLE_LETTERS = 10
/** Numbers drawn per character (the first ones of its vector). */
export const SHOWN_DIMS = 16
/** Hidden MLP neurons drawn for the selected character (the most active ones). */
export const SHOWN_HIDDEN = 24
/** Attention arcs with a smaller weight are not drawn. */
export const ARC_MIN = 0.05
const TOP_OUTPUT = 5

export interface ViewArc {
	/** Index into the visible letters, or "earlier" for everything before them. */
	from: number | "earlier"
	head: number
	weight: number
}

export interface ViewBlock {
	/** Attention into the selected letter, strongest first. */
	arcs: ViewArc[]
	/** The selected letter's numbers after attention. */
	afterAttention: number[]
	/** The selected letter's most active hidden MLP neurons, in neuron order. */
	hidden: { index: number; value: number }[]
	/** Every visible letter's numbers after this block. */
	output: number[][]
	/** What flows along each drawn MLP connection into the hidden layer: [shown input][shown hidden]. */
	inFlows: number[][]
	/** … and out of it: [shown hidden][shown output]. Both empty when no weights were given. */
	outFlows: number[][]
}

/** Each block's MLP weights: W1 [4d × d] and W2 [d × 4d]. */
export type MlpWeights = { W1: Float32Array; W2: Float32Array }[]

export function mlpWeightsOf(net: Transformer): MlpWeights {
	const names = paramShapes(net.config).map((p) => p.name)
	const params = net.params()
	return Array.from({ length: net.config.blocks }, (_, b) => ({
		W1: params[names.indexOf(`b${b}.W1`)],
		W2: params[names.indexOf(`b${b}.W2`)],
	}))
}

/** Everything the transformer view draws for one forward pass (batch 1). */
export interface TextViewModel {
	/** Window position of the first visible letter. */
	first: number
	/** Display strings of the visible letters. */
	letters: string[]
	/** Index into `letters` of the selected letter. */
	selected: number
	/** Whether the selected letter is the last one, i.e. the one whose next character gets written. */
	isNewest: boolean
	embedding: number[][]
	blocks: ViewBlock[]
	/** Most likely characters after the selected letter, most likely first. */
	top: { id: number; char: string; p: number }[]
	config: TransformerConfig
}

const firstDims = (stream: Float32Array, position: number, d: number) =>
	Array.from(stream.subarray(position * d, position * d + Math.min(SHOWN_DIMS, d)))

/**
 * labels: display string per vocabulary id. selected: window position of the selected character
 * (clamped to the visible ones). head: draw only this head's arcs, or null for all heads.
 */
export function buildTextViewModel(
	fwd: TransformerForward,
	config: TransformerConfig,
	labels: readonly string[],
	selected: number,
	head: number | null,
	weights?: MlpWeights,
): TextViewModel {
	const { T } = fwd
	const d = config.dModel
	const F = 4 * d
	const first = Math.max(0, T - VISIBLE_LETTERS)
	const sel = Math.min(T - 1, Math.max(first, selected))
	const visible = Array.from({ length: T - first }, (_, i) => first + i)

	const blocks = fwd.attention.map((att, b): ViewBlock => {
		const arcs: ViewArc[] = []
		for (let h = 0; h < config.heads; h++) {
			if (head !== null && h !== head) continue
			const row = (h * T + sel) * T
			let earlier = 0
			for (let s = 0; s < first; s++) earlier += att[row + s]
			if (earlier >= ARC_MIN) arcs.push({ from: "earlier", head: h, weight: earlier })
			for (let s = first; s <= sel; s++) {
				if (att[row + s] >= ARC_MIN) arcs.push({ from: s - first, head: h, weight: att[row + s] })
			}
		}
		arcs.sort((a, b2) => b2.weight - a.weight)
		const hiddenRow = fwd.mlpHidden[b].subarray(sel * F, (sel + 1) * F)
		const hidden = Array.from(hiddenRow, (value, index) => ({ index, value }))
			.sort((a, b2) => Math.abs(b2.value) - Math.abs(a.value))
			.slice(0, Math.min(SHOWN_HIDDEN, F))
			.sort((a, b2) => a.index - b2.index)
		const shown = Math.min(SHOWN_DIMS, d)
		const w = weights?.[b]
		// flow = the value leaving a neuron × the weight of the connection (as in the digit view)
		const inFlows = w
			? Array.from({ length: shown }, (_, k) => hidden.map((h) => fwd.mlpInput[b][sel * d + k] * w.W1[h.index * d + k]))
			: []
		const outFlows = w
			? hidden.map((h) => Array.from({ length: shown }, (_, k) => h.value * w.W2[k * F + h.index]))
			: []
		return {
			arcs,
			afterAttention: firstDims(fwd.afterAttention[b], sel, d),
			hidden,
			output: visible.map((p) => firstDims(fwd.streams[b + 1], p, d)),
			inFlows,
			outFlows,
		}
	})

	const V = config.vocab
	const top = Array.from({ length: V }, (_, id) => ({ id, char: labels[id], p: fwd.probs[sel * V + id] }))
		.sort((a, b) => b.p - a.p)
		.slice(0, TOP_OUTPUT)

	return {
		first,
		letters: visible.map((p) => labels[fwd.ids[p]]),
		selected: sel - first,
		isNewest: sel === T - 1,
		embedding: visible.map((p) => firstDims(fwd.streams[0], p, d)),
		blocks,
		top,
		config,
	}
}

const pct = (p: number) => `${Math.round(p * 100)}%`

function describeArc(vm: TextViewModel, arc: ViewArc): string {
	const what = arc.from === "earlier" ? "earlier text" : `"${vm.letters[arc.from]}"`
	return `**${what} ${pct(arc.weight)}** (head ${arc.head + 1})`
}

function strongest(vm: TextViewModel, block: number, count: number): string {
	const arcs = vm.blocks[block]?.arcs ?? []
	if (arcs.length === 0) return `it spreads its attention thinly (nothing above ${pct(ARC_MIN)})`
	return arcs
		.slice(0, count)
		.map((a) => describeArc(vm, a))
		.join(" and ")
}

/**
 * Explanation of one player step with the real values. Bold parts are marked **like this**.
 * pick: the sampled next character (step ⑦), or null if none was drawn yet.
 */
export function caption(
	step: number,
	vm: TextViewModel,
	pick: { char: string; p: number } | null,
	temperature: number,
): string {
	const { dModel: d, context, vocab } = vm.config
	const cur = `"${vm.letters[vm.selected]}"`
	switch (step) {
		case 0:
			return `The model reads the text one character at a time. Now it predicts what comes after **${cur}**, and it may use up to ${context} characters before it.`
		case 1:
			return `A computer can't calculate with letters, so each character becomes a learned list of **${d} numbers** (its neurons), plus a pattern that marks its position. ${Math.min(SHOWN_DIMS, d)} of the ${d} are drawn.`
		case 2:
			return `**Attention.** ${cur} looks back at the characters before it and decides how much each one matters right now. Strongest: ${strongest(vm, 0, 2)}. A weighted mix of their neurons is added to its own.`
		case 3:
			return `**MLP.** Each character's neurons now go through a small classic network, like the digit MLP: **${d} → ${4 * d} → ${d}** neurons, with the same weights for every character. The ${Math.min(SHOWN_HIDDEN, 4 * d)} most active middle neurons are drawn; a connection's color shows what flows through it: **orange** pushes the next neuron up, **blue** pushes it down.`
		case 4:
			return vm.blocks.length > 1
				? `**Block 2** repeats attention and MLP on block 1's result, so the model can build on what it already worked out. Its strongest look: ${strongest(vm, 1, 1)}.`
				: "This model has a single block, so the result goes straight to the output."
		case 5: {
			const top = vm.top
				.slice(0, 3)
				.map((t, i) => (i === 0 ? `**"${t.char}" ${pct(t.p)}**` : `"${t.char}" ${pct(t.p)}`))
				.join(", ")
			const inner = vm.isNewest
				? ""
				: " Every position predicts its own next character like this; during training all of them are checked against the real text at once."
			return `**Output.** One output neuron per possible character (${vocab} of them). Softmax turns them into probabilities: ${top}.${inner}`
		}
		default: {
			const how = `drawn at random, weighted by those probabilities (temperature ${temperature.toFixed(2)}: lower = safer, higher = more adventurous)`
			if (!pick) return `The next character is ${how}.`
			const picked = `Picked **"${pick.char}"** (${pct(pick.p)}).`
			return vm.isNewest
				? `The next character is ${how}. ${picked} It is added to the text and everything starts again at ①.`
				: `The next character is ${how}. ${picked} This position isn't the end of the text, so nothing is added; it shows what the model would write here.`
		}
	}
}

/** "a **b** c" → plain and bold parts, for building DOM nodes without innerHTML. */
export function captionParts(text: string): { text: string; bold: boolean }[] {
	return text
		.split("**")
		.map((part, i) => ({ text: part, bold: i % 2 === 1 }))
		.filter((p) => p.text !== "")
}
