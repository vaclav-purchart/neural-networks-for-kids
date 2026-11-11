import { describe, expect, it } from "vitest"
import { crossEntropy, relu, softmaxRows } from "../../src/nn/activations"

describe("activations", () => {
	it("relu zeroes negatives", () => {
		const out = new Float32Array(3)
		relu(Float32Array.from([-1, 0, 2]), out)
		expect([...out]).toEqual([0, 0, 2])
	})

	it("softmax rows sum to one and match known values", () => {
		const out = new Float32Array(4)
		softmaxRows(Float32Array.from([0, Math.log(2), 1000, 1000]), out, 2, 2)
		expect(out[0]).toBeCloseTo(1 / 3, 6)
		expect(out[1]).toBeCloseTo(2 / 3, 6)
		expect(out[2]).toBeCloseTo(0.5, 6) // large logits must not overflow
		expect(out[3]).toBeCloseTo(0.5, 6)
	})

	it("cross-entropy is the mean negative log-probability of the true class", () => {
		const probs = Float32Array.from([0.25, 0.75, 0.5, 0.5])
		expect(crossEntropy(probs, [1, 0], 2, 2)).toBeCloseTo((-Math.log(0.75) - Math.log(0.5)) / 2, 6)
	})
})
