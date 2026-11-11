import { describe, expect, it } from "vitest"
import { Adam } from "../../src/nn/adam"
import { buildNetwork, cnnSpecs, mlpSpecs } from "../../src/nn/architectures"
import type { Network } from "../../src/nn/network"
import { mulberry32 } from "../../src/nn/rng"
import { describeShape, vectorShape } from "../../src/nn/tensor-shape"

/** Compare backprop with central differences for every parameter. */
function expectGradientsMatch(net: Network, x: Float32Array, batch: number, labels: number[]): void {
	const grads = net.backward(net.forward(x, batch), labels)
	const lossAt = () => net.loss(net.forward(x, batch), labels)
	const eps = 1e-3
	net.params().forEach((param, p) => {
		for (let i = 0; i < param.length; i++) {
			const orig = param[i]
			param[i] = orig + eps
			const plus = lossAt()
			param[i] = orig - eps
			const minus = lossAt()
			param[i] = orig
			const numeric = (plus - minus) / (2 * eps)
			const tolerance = 2e-3 + 2e-2 * Math.max(Math.abs(numeric), Math.abs(grads[p][i]))
			expect(Math.abs(numeric - grads[p][i])).toBeLessThan(tolerance)
		}
	})
}

/** True when no pool window has its top two (positive) values closer than `gap`. */
function poolWindowsSeparated(
	act: Float32Array,
	batch: number,
	channels: number,
	h: number,
	w: number,
	gap: number,
): boolean {
	for (let n = 0; n < batch; n++) {
		for (let c = 0; c < channels; c++) {
			for (let oy = 0; oy + 1 < h; oy += 2) {
				for (let ox = 0; ox + 1 < w; ox += 2) {
					const base = ((n * channels + c) * h + oy) * w + ox
					const v = [act[base], act[base + 1], act[base + w], act[base + w + 1]].sort((a, b) => b - a)
					if (v[0] > 0 && v[0] - v[1] < gap) return false
				}
			}
		}
	}
	return true
}

function randomizeBiases(net: Network, rand: () => number): void {
	for (const layer of net.layers) {
		const b = layer.params[1]
		if (b) for (let i = 0; i < b.length; i++) b[i] = (rand() - 0.5) * 0.2
	}
}

describe("Network (MLP)", () => {
	it("backward matches numerical gradients", () => {
		// Same seed and draw order as the original Mlp test: no hidden pre-activation near the ReLU kink.
		const rand = mulberry32(4)
		const net = buildNetwork(vectorShape(4), mlpSpecs([5], 3))
		net.init(rand)
		randomizeBiases(net, rand)
		const batch = 3
		const x = Float32Array.from({ length: batch * 4 }, () => rand() * 2 - 1)
		expect(Math.min(...Array.from(net.forward(x, batch).pre[0], Math.abs))).toBeGreaterThan(0.05)
		expectGradientsMatch(net, x, batch, [0, 2, 1])
	})

	it("learns XOR", () => {
		const net = buildNetwork(vectorShape(2), mlpSpecs([16], 2))
		net.init(mulberry32(7))
		const adam = new Adam(net.params(), { lr: 0.05 })
		const x = Float32Array.from([0, 0, 0, 1, 1, 0, 1, 1])
		const y = [0, 1, 1, 0]
		for (let step = 0; step < 300; step++) adam.step(net.backward(net.forward(x, 4), y))
		const probs = net.forward(x, 4).act[2]
		expect(y.map((_, n) => (probs[n * 2 + 1] > probs[n * 2] ? 1 : 0))).toEqual(y)
	})

	it("rejects input of the wrong length and mismatched parameters", () => {
		const net = buildNetwork(vectorShape(3), mlpSpecs([], 2))
		expect(() => net.forward(new Float32Array(4), 1)).toThrow(/Input length/)
		expect(() => net.setParams([new Float32Array(5), new Float32Array(2)])).toThrow(/wrong shape/)
	})
})

describe("Network (CNN)", () => {
	/** First seed where finite differences are meaningful: no pre-activation near a ReLU kink, no near-tie in a pool window. */
	function wellConditionedCnn(): { net: Network; x: Float32Array } {
		for (let seed = 1; seed < 500; seed++) {
			const rand = mulberry32(seed)
			const net = buildNetwork({ c: 1, h: 6, w: 6 }, [
				{ type: "conv2d", filters: 2, kernel: 3, activation: "relu" },
				{ type: "maxpool", size: 2 },
				{ type: "dense", units: 3, activation: "softmax" },
			])
			net.init(rand)
			randomizeBiases(net, rand)
			const x = Float32Array.from({ length: 2 * 36 }, () => rand() * 2 - 1)
			const fwd = net.forward(x, 2)
			if (fwd.pre[0].every((v) => Math.abs(v) > 0.05) && poolWindowsSeparated(fwd.act[1], 2, 2, 4, 4, 0.05))
				return { net, x }
		}
		throw new Error("no well-conditioned seed found")
	}

	it("backward matches numerical gradients through conv and max-pool", () => {
		const { net, x } = wellConditionedCnn()
		expectGradientsMatch(net, x, 2, [0, 2])
	})

	it("builds the default CNN with the expected shapes", () => {
		const net = buildNetwork({ c: 1, h: 28, w: 28 }, cnnSpecs([8, 16], [64], 10))
		expect(net.layers.map((l) => describeShape(l.outShape))).toEqual([
			"8×26×26",
			"8×13×13",
			"16×11×11",
			"16×5×5",
			"64",
			"10",
		])
		expect(net.params().reduce((sum, p) => sum + p.length, 0)).toBe(27_562)
	})

	it("explains impossible architectures", () => {
		expect(() => buildNetwork({ c: 1, h: 28, w: 28 }, cnnSpecs([8, 8, 8, 8], [64], 10))).toThrow(
			/layer 6 \(conv2d\): input 8×1×1 is smaller than the 3×3 kernel/,
		)
		expect(() =>
			buildNetwork({ c: 1, h: 6, w: 6 }, [
				{ type: "dense", units: 4, activation: "relu" },
				{ type: "conv2d", filters: 2, kernel: 3, activation: "relu" },
				{ type: "dense", units: 2, activation: "softmax" },
			]),
		).toThrow(/layer 1 \(conv2d\): conv2d\/maxpool layers must come before dense layers/)
		expect(() => buildNetwork(vectorShape(3), [{ type: "dense", units: 2, activation: "relu" }])).toThrow(
			/last layer must be dense with softmax/,
		)
	})
})
