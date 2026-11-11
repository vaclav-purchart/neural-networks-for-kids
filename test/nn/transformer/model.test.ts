import { describe, expect, it } from "vitest"
import { Adam } from "../../../src/nn/adam"
import { linearForward } from "../../../src/nn/matrix"
import { mulberry32 } from "../../../src/nn/rng"
import {
	paramShapes,
	Transformer,
	type TransformerConfig,
	transformerParamSizes,
	validateConfig,
} from "../../../src/nn/transformer/model"
import { expectGradients } from "./gradcheck"

const tiny: TransformerConfig = { vocab: 5, context: 4, dModel: 8, heads: 2, blocks: 2 }

function randomIds(n: number, vocab: number, seed: number): number[] {
	const rand = mulberry32(seed)
	return Array.from({ length: n }, () => Math.floor(rand() * vocab))
}

describe("Transformer config", () => {
	it("lists parameters in the documented order", () => {
		const names = paramShapes({ ...tiny, blocks: 1 }).map((p) => p.name)
		expect(names).toEqual([
			"tok",
			"pos",
			"b0.ln1.g",
			"b0.ln1.b",
			"b0.Wq",
			"b0.bq",
			"b0.Wk",
			"b0.bk",
			"b0.Wv",
			"b0.bv",
			"b0.Wo",
			"b0.bo",
			"b0.ln2.g",
			"b0.ln2.b",
			"b0.W1",
			"b0.b1",
			"b0.W2",
			"b0.b2",
			"lnf.g",
			"lnf.b",
			"Wout",
			"bout",
		])
	})

	it("has ~110k weights with the defaults", () => {
		const total = transformerParamSizes({ vocab: 65, context: 64, dModel: 64, heads: 4, blocks: 2 }).reduce(
			(a, b) => a + b,
		)
		expect(total).toBeGreaterThan(100_000)
		expect(total).toBeLessThan(120_000)
	})

	it("rejects a model width that the heads don't divide", () => {
		expect(() => validateConfig({ ...tiny, dModel: 10, heads: 4 })).toThrow(/divisible/)
		expect(() => validateConfig({ ...tiny, blocks: 0 })).toThrow(/blocks/)
	})
})

describe("Transformer", () => {
	it("initializes gains to 1, biases to 0 and weights small", () => {
		const model = new Transformer(tiny)
		model.init(mulberry32(1))
		const byName = Object.fromEntries(paramShapes(tiny).map((p, i) => [p.name, model.params()[i]]))
		expect([...byName["b0.ln1.g"]].every((v) => v === 1)).toBe(true)
		expect([...byName["b1.bq"]].every((v) => v === 0)).toBe(true)
		expect([...byName.bout].every((v) => v === 0)).toBe(true)
		expect(byName.tok.some((v) => v !== 0)).toBe(true)
		expect(Math.max(...byName.Wout.map(Math.abs))).toBeLessThan(0.15)
	})

	it("backward matches numerical gradients for every parameter", () => {
		const model = new Transformer(tiny)
		model.init(mulberry32(3))
		// Non-trivial LayerNorm gains/biases so their gradients are exercised.
		const rand = mulberry32(4)
		for (const p of model.params()) for (let i = 0; i < p.length; i++) p[i] += (rand() - 0.5) * 0.2
		const batch = 2
		const T = 4
		const ids = randomIds(batch * T, tiny.vocab, 5)
		const targets = randomIds(batch * T, tiny.vocab, 6)
		const grads = model.backward(model.forward(ids, batch, T), targets)
		expectGradients(model.params(), grads, () => model.loss(model.forward(ids, batch, T), targets))
	})

	it("never looks ahead: changing a later character leaves earlier outputs unchanged", () => {
		const model = new Transformer(tiny)
		model.init(mulberry32(8))
		const a = [1, 2, 3, 4]
		const b = [1, 2, 0, 0]
		const pa = model.forward(a, 1, 4).probs
		const pb = model.forward(b, 1, 4).probs
		for (let i = 0; i < 2 * tiny.vocab; i++) expect(pb[i]).toBe(pa[i])
		expect(pb[2 * tiny.vocab]).not.toBe(pa[2 * tiny.vocab])
	})

	it("accepts sequences shorter than the context", () => {
		const model = new Transformer(tiny)
		model.init(mulberry32(1))
		const fwd = model.forward([3], 1, 1)
		expect(fwd.probs.length).toBe(tiny.vocab)
		expect(fwd.attention[0].length).toBe(tiny.heads)
	})

	it("rejects sequences longer than the context and unknown token ids", () => {
		const model = new Transformer(tiny)
		expect(() => model.forward([0, 1, 2, 3, 4], 1, 5)).toThrow(/context/)
		expect(() => model.forward([0, 9], 1, 2)).toThrow(/token id 9/)
	})

	it("works with a large vocabulary", () => {
		const model = new Transformer({ vocab: 120, context: 8, dModel: 16, heads: 4, blocks: 1 })
		model.init(mulberry32(2))
		const probs = model.forward([0, 119, 60], 1, 3).probs
		expect(probs.length).toBe(3 * 120)
		expect(probs.subarray(240).reduce((s, p) => s + p, 0)).toBeCloseTo(1, 5)
	})

	it("learns a repeating pattern", () => {
		const config: TransformerConfig = { vocab: 3, context: 8, dModel: 16, heads: 2, blocks: 1 }
		const model = new Transformer(config)
		model.init(mulberry32(11))
		const adam = new Adam(model.params(), { lr: 1e-2 })
		const text = Array.from({ length: 64 }, (_, i) => i % 3)
		const batch = 4
		const T = 8
		const x = new Int32Array(batch * T)
		const y = new Int32Array(batch * T)
		let loss = Infinity
		for (let step = 0; step < 300; step++) {
			for (let n = 0; n < batch; n++) {
				const start = (step * 7 + n * 5) % (text.length - T - 1)
				for (let t = 0; t < T; t++) {
					x[n * T + t] = text[start + t]
					y[n * T + t] = text[start + t + 1]
				}
			}
			const fwd = model.forward(x, batch, T)
			loss = model.loss(fwd, y)
			adam.step(model.backward(fwd, y))
		}
		expect(loss).toBeLessThan(0.05)
	})

	it("setParams copies values and checks sizes", () => {
		const model = new Transformer(tiny)
		const sizes = transformerParamSizes(tiny)
		model.setParams(sizes.map((n) => new Float32Array(n).fill(0.5)))
		expect(model.params()[0][0]).toBe(0.5)
		expect(() => model.setParams([new Float32Array(1)])).toThrow(/expected/)
	})

	it("exposes each block's attention result and MLP hidden layer", () => {
		const model = new Transformer(tiny)
		model.init(mulberry32(5))
		const T = 3
		const fwd = model.forward([1, 2, 3], 1, T)
		const d = tiny.dModel
		const F = 4 * d
		const names = paramShapes(tiny).map((p) => p.name)
		for (let b = 0; b < tiny.blocks; b++) {
			expect(fwd.afterAttention[b]).toHaveLength(T * d)
			expect(fwd.mlpHidden[b]).toHaveLength(T * F)
			// block output = after-attention stream + W2 · hidden + b2
			const W2 = model.params()[names.indexOf(`b${b}.W2`)]
			const b2 = model.params()[names.indexOf(`b${b}.b2`)]
			const mlpOut = new Float32Array(T * d)
			linearForward(fwd.mlpHidden[b], W2, b2, mlpOut, T, F, d)
			for (let i = 0; i < T * d; i++) {
				expect(fwd.streams[b + 1][i]).toBeCloseTo(fwd.afterAttention[b][i] + mlpOut[i], 5)
			}
		}
	})

	it("exposes each block's MLP input, which W1 turns into the hidden layer", () => {
		const model = new Transformer(tiny)
		model.init(mulberry32(6))
		const T = 2
		const fwd = model.forward([4, 1], 1, T)
		const d = tiny.dModel
		const F = 4 * d
		const names = paramShapes(tiny).map((p) => p.name)
		const W1 = model.params()[names.indexOf("b0.W1")]
		const b1 = model.params()[names.indexOf("b0.b1")]
		const pre = new Float32Array(T * F)
		linearForward(fwd.mlpInput[0], W1, b1, pre, T, d, F)
		const gelu = (x: number) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)))
		for (let i = 0; i < T * F; i++) expect(fwd.mlpHidden[0][i]).toBeCloseTo(gelu(pre[i]), 5)
	})
})
