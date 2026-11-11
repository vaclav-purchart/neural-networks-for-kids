import { describe, expect, it } from "vitest"
import { mulberry32 } from "../../src/nn/rng"
import { Transformer } from "../../src/nn/transformer/model"
import { vocabFromChars } from "../../src/nn/transformer/tokenizer"
import { evaluateText, evalWindows, fillBatch, lrAt, promptIds, snapshotSteps } from "../../src/trainer/text-training"

describe("snapshotSteps", () => {
	it("is dense early, evenly spaced later, and ends at the last step", () => {
		const s = snapshotSteps(1200)
		expect(s.slice(0, 6)).toEqual([0, 10, 25, 50, 100, 200])
		expect(s.at(-1)).toBe(1200)
		expect(new Set(s).size).toBe(s.length)
		expect([...s].sort((a, b) => a - b)).toEqual(s)
		expect(s.length).toBeGreaterThanOrEqual(15)
		expect(s.length).toBeLessThanOrEqual(25)
	})
	it("handles tiny runs", () => {
		expect(snapshotSteps(5)).toEqual([0, 5])
	})
})

describe("lrAt", () => {
	it("decays linearly from the base rate to 10%", () => {
		expect(lrAt(0, 100, 1)).toBe(1)
		expect(lrAt(50, 100, 1)).toBeCloseTo(0.55)
		expect(lrAt(100, 100, 1)).toBeCloseTo(0.1)
	})
})

describe("fillBatch", () => {
	it("copies windows and their next-character targets", () => {
		const ids = Uint16Array.from([0, 1, 2, 3, 4, 5, 6])
		const x = new Int32Array(6)
		const y = new Int32Array(6)
		fillBatch(ids, [0, 3], 3, x, y)
		expect([...x]).toEqual([0, 1, 2, 3, 4, 5])
		expect([...y]).toEqual([1, 2, 3, 4, 5, 6])
	})
})

describe("evalWindows", () => {
	it("spreads windows evenly and keeps them inside the text", () => {
		const w = evalWindows(100, 9, 4)
		expect(w).toHaveLength(4)
		for (const s of w) expect(s + 9).toBeLessThanOrEqual(99)
		expect(evalWindows(12, 9, 10)).toEqual([0, 1, 2])
	})
})

describe("evaluateText", () => {
	it("reports loss and accuracy over every position of the windows", () => {
		const model = new Transformer({ vocab: 3, context: 4, dModel: 8, heads: 2, blocks: 1 })
		model.init(mulberry32(1))
		const ids = Uint16Array.from({ length: 40 }, (_, i) => i % 3)
		const r = evaluateText(model, ids, [0, 5, 10], 4)
		expect(r.samples).toBe(12)
		expect(r.loss).toBeGreaterThan(0)
		expect(r.accuracy).toBeGreaterThanOrEqual(0)
		expect(r.accuracy).toBeLessThanOrEqual(1)
	})
})

describe("promptIds", () => {
	it("takes the first line of the test text including its newline, at most 32 chars", () => {
		const vocab = vocabFromChars(["\n", "a", "b"])
		expect(promptIds(Uint16Array.from([1, 2, 0, 1, 1]), vocab)).toEqual([1, 2, 0])
		expect(promptIds(Uint16Array.from(new Array(50).fill(1)), vocab)).toHaveLength(32)
		expect(promptIds(Uint16Array.from([0, 1]), vocab)).toEqual([0])
	})
})
