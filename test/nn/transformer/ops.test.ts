import { describe, expect, it } from "vitest"
import { mulberry32, randomNormal } from "../../../src/nn/rng"
import {
	attentionBackward,
	attentionForward,
	embeddingBackward,
	embeddingForward,
	gelu,
	geluBackward,
	layerNormBackward,
	layerNormForward,
} from "../../../src/nn/transformer/ops"
import { expectGradients } from "./gradcheck"

const rand = mulberry32(7)
const randn = (n: number, scale = 1) => Float32Array.from({ length: n }, () => randomNormal(rand) * scale)
const dot = (a: Float32Array, b: Float32Array) => a.reduce((s, v, i) => s + v * b[i], 0)

describe("embedding", () => {
	it("adds token and position vectors, and backprops into both", () => {
		const [batch, T, d, V] = [2, 3, 4, 5]
		const ids = [0, 4, 4, 2, 1, 0]
		const tok = randn(V * d)
		const pos = randn(T * d)
		const out = new Float32Array(batch * T * d)
		embeddingForward(ids, tok, pos, out, batch, T, d)
		expect(out[(1 * T + 2) * d + 3]).toBeCloseTo(tok[0 * d + 3] + pos[2 * d + 3], 6)
		const R = randn(out.length)
		const dTok = new Float32Array(tok.length)
		const dPos = new Float32Array(pos.length)
		embeddingBackward(ids, R, dTok, dPos, batch, T, d)
		expectGradients([tok, pos], [dTok, dPos], () => {
			embeddingForward(ids, tok, pos, out, batch, T, d)
			return dot(out, R)
		})
	})
})

describe("layerNorm", () => {
	it("normalizes each row to mean 0 / variance 1 before gain and bias", () => {
		const [rows, d] = [3, 6]
		const x = randn(rows * d, 3)
		const y = new Float32Array(rows * d)
		layerNormForward(x, new Float32Array(d).fill(1), new Float32Array(d), y, rows, d)
		const row = y.subarray(d, 2 * d)
		const mean = row.reduce((s, v) => s + v, 0) / d
		const variance = row.reduce((s, v) => s + (v - mean) ** 2, 0) / d
		expect(mean).toBeCloseTo(0, 5)
		expect(variance).toBeCloseTo(1, 3)
	})

	it("backward matches numerical gradients for x, gain and bias", () => {
		const [rows, d] = [3, 5]
		const x = randn(rows * d)
		const g = randn(d)
		const b = randn(d)
		const y = new Float32Array(rows * d)
		const R = randn(rows * d)
		const cache = layerNormForward(x, g, b, y, rows, d)
		const dx = new Float32Array(x.length)
		const dg = new Float32Array(d)
		const db = new Float32Array(d)
		layerNormBackward(R, cache, g, dx, dg, db, rows, d)
		expectGradients([x, g, b], [dx, dg, db], () => {
			layerNormForward(x, g, b, y, rows, d)
			return dot(y, R)
		})
	})
})

describe("gelu", () => {
	it("is ~0 for large negative, ~x for large positive, and 0 at 0", () => {
		const y = new Float32Array(3)
		gelu(Float32Array.from([-6, 0, 6]), y)
		expect(y[0]).toBeCloseTo(0, 4)
		expect(y[1]).toBe(0)
		expect(y[2]).toBeCloseTo(6, 4)
	})

	it("backward matches numerical gradients", () => {
		const x = randn(20, 2)
		const y = new Float32Array(20)
		const R = randn(20)
		const dx = new Float32Array(20)
		geluBackward(x, R, dx)
		expectGradients([x], [dx], () => {
			gelu(x, y)
			return dot(y, R)
		})
	})
})

describe("causal multi-head attention", () => {
	const [batch, T, d, H] = [2, 4, 6, 2]
	const n = batch * T * d

	it("rows of attention weights sum to 1 and never look ahead", () => {
		const [q, k, v] = [randn(n), randn(n), randn(n)]
		const att = new Float32Array(batch * H * T * T)
		attentionForward(q, k, v, att, new Float32Array(n), batch, T, d, H)
		for (let r = 0; r < batch * H * T; r++) {
			const t = r % T
			const row = att.subarray(r * T, (r + 1) * T)
			expect(row.reduce((s, w) => s + w, 0)).toBeCloseTo(1, 5)
			for (let s = t + 1; s < T; s++) expect(row[s]).toBe(0)
		}
	})

	it("the first position attends only to itself, so its output equals its value", () => {
		const [q, k, v] = [randn(n), randn(n), randn(n)]
		const out = new Float32Array(n)
		attentionForward(q, k, v, new Float32Array(batch * H * T * T), out, batch, T, d, H)
		for (let j = 0; j < d; j++) expect(out[j]).toBeCloseTo(v[j], 5)
	})

	it("backward matches numerical gradients for q, k and v", () => {
		const [q, k, v] = [randn(n), randn(n), randn(n)]
		const att = new Float32Array(batch * H * T * T)
		const out = new Float32Array(n)
		const R = randn(n)
		attentionForward(q, k, v, att, out, batch, T, d, H)
		const [dq, dk, dv] = [new Float32Array(n), new Float32Array(n), new Float32Array(n)]
		attentionBackward(q, k, v, att, R, dq, dk, dv, batch, T, d, H)
		expectGradients([q, k, v], [dq, dk, dv], () => {
			attentionForward(q, k, v, att, out, batch, T, d, H)
			return dot(out, R)
		})
	})
})
