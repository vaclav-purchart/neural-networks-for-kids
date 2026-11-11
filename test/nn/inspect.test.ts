import { describe, expect, it } from "vitest"
import { buildNetwork, mlpSpecs } from "../../src/nn/architectures"
import { inspectFeatureMap, inspectNeuron } from "../../src/nn/inspect"
import type { Conv2d } from "../../src/nn/layers"
import { vectorShape } from "../../src/nn/tensor-shape"

describe("inspectNeuron", () => {
	const net = buildNetwork(vectorShape(2), mlpSpecs([2], 2))
	net.setParams([
		Float32Array.from([1, -2, 0.5, 0.5]),
		Float32Array.from([0.1, 0]),
		Float32Array.from([1, 0, 0, 1]),
		Float32Array.from([0, 0]),
	])
	const fwd = net.forward(Float32Array.from([1, 1]), 1)

	it("breaks a hidden neuron into bias + contributions", () => {
		const d = inspectNeuron(net, fwd, 1, 0)
		expect(d.bias).toBeCloseTo(0.1)
		expect(d.pre).toBeCloseTo(1 - 2 + 0.1)
		expect(d.activation).toBe(0) // ReLU of -0.9
		expect(d.incoming.map((c) => c.index)).toEqual([1, 0]) // sorted by |contribution|
		expect(d.incoming[0].contribution).toBeCloseTo(-2)
		expect(d.outgoing.map((c) => c.index)).toEqual([0, 1])
	})

	it("has no incoming list for the input layer and no outgoing list for the output layer", () => {
		expect(inspectNeuron(net, fwd, 0, 1).incoming).toEqual([])
		expect(inspectNeuron(net, fwd, 2, 0).outgoing).toEqual([])
	})

	it("rejects out-of-range neurons and non-dense layers", () => {
		expect(() => inspectNeuron(net, fwd, 3, 0)).toThrow(RangeError)
		const cnn = buildNetwork({ c: 1, h: 4, w: 4 }, [
			{ type: "conv2d", filters: 2, kernel: 3, activation: "relu" },
			{ type: "dense", units: 2, activation: "softmax" },
		])
		const cnnFwd = cnn.forward(new Float32Array(16), 1)
		expect(() => inspectNeuron(cnn, cnnFwd, 1, 0)).toThrow(/not a dense layer/)
	})
})

describe("inspectFeatureMap", () => {
	const cnn = buildNetwork({ c: 1, h: 4, w: 4 }, [
		{ type: "conv2d", filters: 2, kernel: 3, activation: "relu" },
		{ type: "dense", units: 2, activation: "softmax" },
	])
	const conv = cnn.layers[0] as Conv2d
	conv.W.set([...Array(9).fill(0), 0, 0, 0, 0, 1, 0, 0, 0, 0]) // filter 1 copies the window's centre pixel
	conv.b.set([0, 0.5])
	const x = Float32Array.from({ length: 16 }, (_, i) => i / 10)
	const fwd = cnn.forward(x, 1)

	it("reports the filter, bias and where the map fires most", () => {
		const d = inspectFeatureMap(cnn, fwd, 1, 1)
		expect(d.kernels).toHaveLength(1)
		expect([...d.kernels[0]]).toEqual([0, 0, 0, 0, 1, 0, 0, 0, 0])
		expect(d.bias).toBe(0.5)
		expect([d.height, d.width]).toEqual([2, 2])
		// outputs: x(1,1)+0.5 = 1.0, x(1,2)+0.5 = 1.1, x(2,1)+0.5 = 1.4, x(2,2)+0.5 = 1.5
		expect(d.maxAt).toEqual([1, 1])
		expect(d.maxValue).toBeCloseTo(1.5)
		expect(d.mean).toBeCloseTo(1.25)
	})

	it("rejects layers that are not convolutions", () => {
		expect(() => inspectFeatureMap(cnn, fwd, 2, 0)).toThrow(RangeError)
		expect(() => inspectFeatureMap(cnn, fwd, 1, 5)).toThrow(RangeError)
	})
})
