import { crossEntropy, softmaxRows } from "../activations"
import { linearForward, linearInputGrad, linearWeightGrad } from "../matrix"
import { type Rand, randomNormal } from "../rng"
import {
	addInto,
	attentionBackward,
	attentionForward,
	embeddingBackward,
	embeddingForward,
	gelu,
	geluBackward,
	type LayerNormCache,
	layerNormBackward,
	layerNormForward,
} from "./ops"

/** A GPT-style decoder: token + position embedding → blocks of (attention, MLP) → softmax over the vocabulary. */
export interface TransformerConfig {
	/** Number of distinct characters. */
	vocab: number
	/** Longest sequence the model sees (T). */
	context: number
	/** Width of every position's vector (d). */
	dModel: number
	/** Attention heads per block (H); dModel / heads is each head's size. */
	heads: number
	/** Number of transformer blocks (N). */
	blocks: number
}

const PER_BLOCK = 16
const BLOCK_NAMES = [
	"ln1.g",
	"ln1.b",
	"Wq",
	"bq",
	"Wk",
	"bk",
	"Wv",
	"bv",
	"Wo",
	"bo",
	"ln2.g",
	"ln2.b",
	"W1",
	"b1",
	"W2",
	"b2",
]

export function validateConfig(c: TransformerConfig): void {
	const positive = (v: number, name: string) => {
		if (!Number.isInteger(v) || v <= 0) throw new Error(`${name} must be a positive integer, got ${v}`)
	}
	positive(c.vocab, "vocab")
	positive(c.context, "context")
	positive(c.dModel, "dModel")
	positive(c.heads, "heads")
	positive(c.blocks, "blocks")
	if (c.vocab < 2) throw new Error("vocab must have at least 2 characters")
	if (c.dModel % c.heads !== 0) throw new Error(`dModel ${c.dModel} must be divisible by heads ${c.heads}`)
}

/** Every learnable tensor in snapshot order; weight matrices are [outDim × inDim]. */
export function paramShapes(c: TransformerConfig): { name: string; size: number }[] {
	const { vocab: V, context: T, dModel: d } = c
	const F = 4 * d
	const block = [d, d, d * d, d, d * d, d, d * d, d, d * d, d, d, d, F * d, F, d * F, d]
	const shapes = [
		{ name: "tok", size: V * d },
		{ name: "pos", size: T * d },
	]
	for (let b = 0; b < c.blocks; b++) {
		block.forEach((size, i) => {
			shapes.push({ name: `b${b}.${BLOCK_NAMES[i]}`, size })
		})
	}
	shapes.push(
		{ name: "lnf.g", size: d },
		{ name: "lnf.b", size: d },
		{ name: "Wout", size: V * d },
		{ name: "bout", size: V },
	)
	return shapes
}

export function transformerParamSizes(c: TransformerConfig): number[] {
	return paramShapes(c).map((p) => p.size)
}

interface BlockCache {
	/** Residual stream entering the block. */
	xIn: Float32Array
	ln1: LayerNormCache
	a: Float32Array
	q: Float32Array
	k: Float32Array
	v: Float32Array
	attOut: Float32Array
	/** Stream after the attention residual add. */
	x1: Float32Array
	ln2: LayerNormCache
	m: Float32Array
	/** MLP hidden layer before / after GELU. */
	h: Float32Array
	hAct: Float32Array
}

export interface TransformerForward {
	batch: number
	T: number
	ids: Int32Array
	/** streams[0] = embeddings, streams[b + 1] = residual stream after block b; each [batch·T × d]. */
	streams: Float32Array[]
	/** attention[b]: [batch × H × T × T] weights of block b. */
	attention: Float32Array[]
	/** afterAttention[b]: residual stream after block b's attention, before its MLP; [batch·T × d]. */
	afterAttention: Float32Array[]
	/** mlpInput[b]: what block b's MLP reads (the stream after LayerNorm); [batch·T × d]. */
	mlpInput: Float32Array[]
	/** mlpHidden[b]: block b's MLP hidden layer after GELU; [batch·T × 4d]. */
	mlpHidden: Float32Array[]
	logits: Float32Array
	probs: Float32Array
	caches: { blocks: BlockCache[]; lnf: LayerNormCache; final: Float32Array }
}

export class Transformer {
	readonly config: TransformerConfig
	private readonly tensors: Float32Array[]

	constructor(config: TransformerConfig) {
		validateConfig(config)
		this.config = { ...config }
		this.tensors = transformerParamSizes(config).map((n) => new Float32Array(n))
	}

	params(): Float32Array[] {
		return this.tensors
	}

	setParams(values: readonly Float32Array[]): void {
		if (values.length !== this.tensors.length) {
			throw new Error(`setParams: expected ${this.tensors.length} tensors, got ${values.length}`)
		}
		values.forEach((v, i) => {
			if (v.length !== this.tensors[i].length) {
				throw new Error(`setParams: tensor ${i} expected ${this.tensors[i].length} values, got ${v.length}`)
			}
			this.tensors[i].set(v)
		})
	}

	/** normal(0, 0.02) weights; residual projections scaled by 1/√(2N); zero biases; LayerNorm gain 1. */
	init(rand: Rand): void {
		const residualStd = 0.02 / Math.sqrt(2 * this.config.blocks)
		paramShapes(this.config).forEach(({ name }, i) => {
			const p = this.tensors[i]
			const leaf = name.slice(name.lastIndexOf(".") + 1)
			if (leaf === "g")
				p.fill(1) // LayerNorm gains
			else if (leaf === "b" || /^b(q|k|v|o|1|2|out)$/.test(leaf))
				p.fill(0) // LayerNorm and dense biases
			else {
				const std = leaf === "Wo" || leaf === "W2" ? residualStd : 0.02
				for (let j = 0; j < p.length; j++) p[j] = randomNormal(rand) * std
			}
		})
	}

	private block(b: number): Float32Array[] {
		return this.tensors.slice(2 + b * PER_BLOCK, 2 + (b + 1) * PER_BLOCK)
	}

	private head(): Float32Array[] {
		return this.tensors.slice(2 + this.config.blocks * PER_BLOCK)
	}

	forward(ids: ArrayLike<number>, batch: number, T: number): TransformerForward {
		const { vocab: V, context, dModel: d, heads: H, blocks: N } = this.config
		if (T > context) throw new Error(`Sequence length ${T} exceeds the context of ${context}`)
		if (ids.length !== batch * T) throw new Error(`Expected ${batch * T} token ids, got ${ids.length}`)
		const tokens = Int32Array.from(ids)
		for (const id of tokens) if (id < 0 || id >= V) throw new Error(`Unknown token id ${id}`)
		const rows = batch * T
		const F = 4 * d
		const x0 = new Float32Array(rows * d)
		embeddingForward(tokens, this.tensors[0], this.tensors[1], x0, batch, T, d)
		const streams = [x0]
		const attention: Float32Array[] = []
		const blocks: BlockCache[] = []
		let x = x0
		for (let b = 0; b < N; b++) {
			const [g1, be1, Wq, bq, Wk, bk, Wv, bv, Wo, bo, g2, be2, W1, b1, W2, b2] = this.block(b)
			const a = new Float32Array(rows * d)
			const ln1 = layerNormForward(x, g1, be1, a, rows, d)
			const q = new Float32Array(rows * d)
			const k = new Float32Array(rows * d)
			const v = new Float32Array(rows * d)
			linearForward(a, Wq, bq, q, rows, d, d)
			linearForward(a, Wk, bk, k, rows, d, d)
			linearForward(a, Wv, bv, v, rows, d, d)
			const att = new Float32Array(batch * H * T * T)
			const attOut = new Float32Array(rows * d)
			attentionForward(q, k, v, att, attOut, batch, T, d, H)
			const x1 = new Float32Array(rows * d)
			linearForward(attOut, Wo, bo, x1, rows, d, d)
			addInto(x1, x)
			const m = new Float32Array(rows * d)
			const ln2 = layerNormForward(x1, g2, be2, m, rows, d)
			const h = new Float32Array(rows * F)
			linearForward(m, W1, b1, h, rows, d, F)
			const hAct = new Float32Array(rows * F)
			gelu(h, hAct)
			const x2 = new Float32Array(rows * d)
			linearForward(hAct, W2, b2, x2, rows, F, d)
			addInto(x2, x1)
			blocks.push({ xIn: x, ln1, a, q, k, v, attOut, x1, ln2, m, h, hAct })
			attention.push(att)
			streams.push(x2)
			x = x2
		}
		const [gf, bf, Wout, bout] = this.head()
		const final = new Float32Array(rows * d)
		const lnf = layerNormForward(x, gf, bf, final, rows, d)
		const logits = new Float32Array(rows * V)
		linearForward(final, Wout, bout, logits, rows, d, V)
		const probs = new Float32Array(rows * V)
		softmaxRows(logits, probs, rows, V)
		return {
			batch,
			T,
			ids: tokens,
			streams,
			attention,
			afterAttention: blocks.map((c) => c.x1),
			mlpInput: blocks.map((c) => c.m),
			mlpHidden: blocks.map((c) => c.hAct),
			logits,
			probs,
			caches: { blocks, lnf, final },
		}
	}

	/** Mean cross-entropy over every position of every sequence. */
	loss(fwd: TransformerForward, targets: ArrayLike<number>): number {
		return crossEntropy(fwd.probs, targets, fwd.batch * fwd.T, this.config.vocab)
	}

	backward(fwd: TransformerForward, targets: ArrayLike<number>): Float32Array[] {
		const { vocab: V, dModel: d, heads: H, blocks: N } = this.config
		const { batch, T } = fwd
		const rows = batch * T
		const F = 4 * d
		const grads = this.tensors.map((p) => new Float32Array(p.length))
		const gradBlock = (b: number) => grads.slice(2 + b * PER_BLOCK, 2 + (b + 1) * PER_BLOCK)
		const [ggf, gbf, gWout, gbout] = grads.slice(2 + N * PER_BLOCK)
		const [gf, , Wout] = this.head()

		// Softmax + cross-entropy: (p − onehot) / rows.
		const dLogits = fwd.probs.slice()
		for (let r = 0; r < rows; r++) dLogits[r * V + targets[r]] -= 1
		for (let i = 0; i < dLogits.length; i++) dLogits[i] /= rows

		linearWeightGrad(dLogits, fwd.caches.final, gWout, gbout, rows, d, V)
		const dFinal = new Float32Array(rows * d)
		linearInputGrad(dLogits, Wout, dFinal, rows, d, V)
		let dx = new Float32Array(rows * d)
		layerNormBackward(dFinal, fwd.caches.lnf, gf, dx, ggf, gbf, rows, d)

		const tmp = new Float32Array(rows * d)
		for (let b = N - 1; b >= 0; b--) {
			const c = fwd.caches.blocks[b]
			const [g1, , Wq, , Wk, , Wv, , Wo, , g2, , W1, , W2] = this.block(b)
			const [gg1, gbe1, gWq, gbq, gWk, gbk, gWv, gbv, gWo, gbo, gg2, gbe2, gW1, gb1, gW2, gb2] = gradBlock(b)

			// x2 = x1 + MLP(LN2(x1))
			const dx1 = dx.slice()
			linearWeightGrad(dx, c.hAct, gW2, gb2, rows, F, d)
			const dhAct = new Float32Array(rows * F)
			linearInputGrad(dx, W2, dhAct, rows, F, d)
			const dh = new Float32Array(rows * F)
			geluBackward(c.h, dhAct, dh)
			linearWeightGrad(dh, c.m, gW1, gb1, rows, d, F)
			const dm = new Float32Array(rows * d)
			linearInputGrad(dh, W1, dm, rows, d, F)
			layerNormBackward(dm, c.ln2, g2, tmp, gg2, gbe2, rows, d)
			addInto(dx1, tmp)

			// x1 = xIn + Wo · attention(LN1(xIn))
			const dxIn = dx1.slice()
			linearWeightGrad(dx1, c.attOut, gWo, gbo, rows, d, d)
			const dAttOut = new Float32Array(rows * d)
			linearInputGrad(dx1, Wo, dAttOut, rows, d, d)
			const dq = new Float32Array(rows * d)
			const dk = new Float32Array(rows * d)
			const dv = new Float32Array(rows * d)
			attentionBackward(c.q, c.k, c.v, fwd.attention[b], dAttOut, dq, dk, dv, batch, T, d, H)
			linearWeightGrad(dq, c.a, gWq, gbq, rows, d, d)
			linearWeightGrad(dk, c.a, gWk, gbk, rows, d, d)
			linearWeightGrad(dv, c.a, gWv, gbv, rows, d, d)
			const da = new Float32Array(rows * d)
			linearInputGrad(dq, Wq, da, rows, d, d)
			linearInputGrad(dk, Wk, tmp, rows, d, d)
			addInto(da, tmp)
			linearInputGrad(dv, Wv, tmp, rows, d, d)
			addInto(da, tmp)
			layerNormBackward(da, c.ln1, g1, tmp, gg1, gbe1, rows, d)
			addInto(dxIn, tmp)
			dx = dxIn
		}
		embeddingBackward(fwd.ids, dx, grads[0], grads[1], batch, T, d)
		return grads
	}
}
