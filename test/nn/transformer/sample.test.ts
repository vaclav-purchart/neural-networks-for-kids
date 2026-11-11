import { describe, expect, it } from "vitest"
import { mulberry32 } from "../../../src/nn/rng"
import { Transformer } from "../../../src/nn/transformer/model"
import { generate, sampleFromLogits, startId } from "../../../src/nn/transformer/sample"
import { vocabFromChars } from "../../../src/nn/transformer/tokenizer"

describe("sampleFromLogits", () => {
	it("picks the argmax at temperature 0", () => {
		expect(sampleFromLogits([0, 1, 5, 2, 9, 0], 3, 3, 0, mulberry32(1))).toBe(1)
	})
	it("follows the probabilities at temperature 1", () => {
		const rand = mulberry32(2)
		const counts = [0, 0]
		for (let i = 0; i < 4000; i++) counts[sampleFromLogits([Math.log(3), 0], 0, 2, 1, rand)]++
		expect(counts[0] / 4000).toBeGreaterThan(0.7)
		expect(counts[0] / 4000).toBeLessThan(0.8)
	})
})

describe("startId", () => {
	it("prefers a newline, else the first character", () => {
		expect(startId(vocabFromChars(["\n", "a"]))).toBe(0)
		expect(startId(vocabFromChars(["a", "b", "\n"]))).toBe(2)
		expect(startId(vocabFromChars(["a", "b"]))).toBe(0)
	})
})

describe("generate", () => {
	const model = new Transformer({ vocab: 4, context: 4, dModel: 8, heads: 2, blocks: 1 })
	model.init(mulberry32(9))

	it("is reproducible for a fixed seed", () => {
		const a = generate(model, [1, 2], 12, 0.8, mulberry32(5), 0)
		const b = generate(model, [1, 2], 12, 0.8, mulberry32(5), 0)
		expect(a).toEqual(b)
		expect(a).toHaveLength(12)
		for (const id of a) expect(id).toBeLessThan(4)
	})

	it("starts from the fallback when the prompt is empty", () => {
		expect(generate(model, [], 3, 0, mulberry32(1), 3)).toEqual(generate(model, [3], 3, 0, mulberry32(1), 0))
	})

	it("uses only the last `context` characters of a long prompt", () => {
		const long = [0, 1, 2, 3, 0, 1, 2, 3, 1, 1]
		expect(generate(model, long, 5, 0, mulberry32(1), 0)).toEqual(
			generate(model, long.slice(-4), 5, 0, mulberry32(1), 0),
		)
	})
})
