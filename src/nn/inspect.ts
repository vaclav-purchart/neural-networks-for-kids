import { Conv2d, Dense } from "./layers"
import type { ForwardResult, Network } from "./network"
import { shapeSize } from "./tensor-shape"

export interface Contribution {
	/** Neuron index in the neighbouring layer. */
	index: number
	/** Incoming: the source activation. Outgoing: the target's activation. */
	value: number
	weight: number
	/** What flows along this connection: source activation × weight. */
	contribution: number
}

export interface NeuronDetails {
	layer: number
	index: number
	activation: number
	pre: number | null
	bias: number | null
	incoming: Contribution[]
	outgoing: Contribution[]
}

/**
 * Explain one neuron for batch item 0. `layer` indexes ForwardResult.act: 0 is the
 * input, k ≥ 1 is the output of layer k−1 (which must be dense). Lists are sorted
 * by |contribution|; outgoing is only listed when the next layer is dense.
 */
export function inspectNeuron(net: Network, fwd: ForwardResult, layer: number, index: number, topK = 8): NeuronDetails {
	const L = net.layers.length
	if (layer < 0 || layer > L) throw new RangeError(`No layer ${layer}`)
	const size = layer === 0 ? net.inputSize : shapeSize(net.layers[layer - 1].outShape)
	if (index < 0 || index >= size) throw new RangeError(`No neuron ${index} in layer ${layer}`)
	const activation = fwd.act[layer][index]
	let pre: number | null = null
	let bias: number | null = null
	const incoming: Contribution[] = []
	if (layer > 0) {
		const dense = net.layers[layer - 1]
		if (!(dense instanceof Dense)) throw new RangeError(`Layer ${layer} is not a dense layer`)
		pre = fwd.pre[layer - 1][index]
		bias = dense.b[index]
		for (let i = 0; i < dense.inDim; i++) {
			const weight = dense.W[index * dense.inDim + i]
			const value = fwd.act[layer - 1][i]
			incoming.push({ index: i, value, weight, contribution: value * weight })
		}
	}
	const outgoing: Contribution[] = []
	const next = net.layers[layer]
	if (next instanceof Dense) {
		for (let o = 0; o < next.outDim; o++) {
			const weight = next.W[o * next.inDim + index]
			outgoing.push({ index: o, value: fwd.act[layer + 1][o], weight, contribution: activation * weight })
		}
	}
	const top = (list: Contribution[]) =>
		list.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution)).slice(0, topK)
	return { layer, index, activation, pre, bias, incoming: top(incoming), outgoing: top(outgoing) }
}

export interface FeatureMapDetails {
	layer: number
	channel: number
	bias: number
	/** One 3×3 kernel (9 weights, row-major) per input channel. */
	kernels: Float32Array[]
	maxValue: number
	/** [row, col] of the strongest activation. */
	maxAt: [number, number]
	mean: number
	height: number
	width: number
}

/** Explain one feature map (channel) of the conv layer whose output is act[layer], for batch item 0. */
export function inspectFeatureMap(net: Network, fwd: ForwardResult, layer: number, channel: number): FeatureMapDetails {
	const conv = net.layers[layer - 1]
	if (!(conv instanceof Conv2d) || channel < 0 || channel >= conv.outShape.c) {
		throw new RangeError(`No feature map ${channel} in layer ${layer}`)
	}
	const { h, w } = conv.outShape
	const C = conv.inShape.c
	const kernels = Array.from({ length: C }, (_, c) => conv.W.slice((channel * C + c) * 9, (channel * C + c + 1) * 9))
	const values = fwd.act[layer].subarray(channel * h * w, (channel + 1) * h * w)
	let maxValue = -Infinity
	let at = 0
	let sum = 0
	values.forEach((v, i) => {
		sum += v
		if (v > maxValue) {
			maxValue = v
			at = i
		}
	})
	return {
		layer,
		channel,
		bias: conv.b[channel],
		kernels,
		maxValue,
		maxAt: [Math.floor(at / w), at % w],
		mean: sum / values.length,
		height: h,
		width: w,
	}
}
