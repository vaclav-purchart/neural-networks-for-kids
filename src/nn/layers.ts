import { linearForward, linearInputGrad, linearWeightGrad } from "./matrix"
import { type Rand, randomNormal } from "./rng"
import { type Shape, shapeSize } from "./tensor-shape"

export type Activation = "relu" | "softmax" | "none"

/** What a layer's forward pass hands to its backward pass. */
export interface LayerForward {
	/** Output before the activation function. */
	pre: Float32Array
	/** Max-pool only: the input element chosen for every output element. */
	argmax?: Int32Array
}

export interface LayerBackward {
	/** Gradient w.r.t. the layer input (empty when not requested). */
	dIn: Float32Array
	/** Gradients in the same order as `params`. */
	grads: Float32Array[]
}

export interface Layer {
	readonly kind: "dense" | "conv2d" | "maxpool"
	readonly inShape: Shape
	readonly outShape: Shape
	readonly activation: Activation
	/** Learnable tensors: [W, b] for dense / conv2d, [] for maxpool. */
	readonly params: Float32Array[]
	/** He-normal weights (good for ReLU), zero biases. */
	init(rand: Rand): void
	forward(x: Float32Array, batch: number): LayerForward
	/** dPre: gradient w.r.t. this layer's pre-activation output. */
	backward(x: Float32Array, dPre: Float32Array, batch: number, fwd: LayerForward, needInputGrad: boolean): LayerBackward
}

/** Always C×H×W (describeShape would print a 1×1 map as a plain vector size). */
const fullShape = (s: Shape): string => `${s.c}×${s.h}×${s.w}`

function heInit(W: Float32Array, fanIn: number, rand: Rand): void {
	const std = Math.sqrt(2 / fanIn)
	for (let i = 0; i < W.length; i++) W[i] = randomNormal(rand) * std
}

/** Fully connected layer. W is row-major [units × inputs]; any input shape is read as a flat vector. */
export class Dense implements Layer {
	readonly kind = "dense" as const
	readonly inShape: Shape
	readonly outShape: Shape
	readonly activation: Activation
	readonly inDim: number
	readonly outDim: number
	readonly W: Float32Array
	readonly b: Float32Array
	readonly params: Float32Array[]

	constructor(inShape: Shape, units: number, activation: Activation) {
		if (!Number.isInteger(units) || units <= 0) throw new Error(`dense needs a positive number of units, got ${units}`)
		this.inShape = inShape
		this.outShape = { c: units, h: 1, w: 1 }
		this.activation = activation
		this.inDim = shapeSize(inShape)
		this.outDim = units
		this.W = new Float32Array(units * this.inDim)
		this.b = new Float32Array(units)
		this.params = [this.W, this.b]
	}

	init(rand: Rand): void {
		heInit(this.W, this.inDim, rand)
	}

	forward(x: Float32Array, batch: number): LayerForward {
		const pre = new Float32Array(batch * this.outDim)
		linearForward(x, this.W, this.b, pre, batch, this.inDim, this.outDim)
		return { pre }
	}

	backward(
		x: Float32Array,
		dPre: Float32Array,
		batch: number,
		_fwd: LayerForward,
		needInputGrad: boolean,
	): LayerBackward {
		const dW = new Float32Array(this.W.length)
		const db = new Float32Array(this.b.length)
		linearWeightGrad(dPre, x, dW, db, batch, this.inDim, this.outDim)
		const dIn = new Float32Array(needInputGrad ? batch * this.inDim : 0)
		if (needInputGrad) linearInputGrad(dPre, this.W, dIn, batch, this.inDim, this.outDim)
		return { dIn, grads: [dW, db] }
	}
}

/** 3×3 convolution, stride 1, no padding. W is [filters × inChannels × 3 × 3]. */
export class Conv2d implements Layer {
	readonly kind = "conv2d" as const
	readonly inShape: Shape
	readonly outShape: Shape
	readonly activation: Activation
	readonly W: Float32Array
	readonly b: Float32Array
	readonly params: Float32Array[]

	constructor(inShape: Shape, filters: number, activation: Activation) {
		if (!Number.isInteger(filters) || filters <= 0)
			throw new Error(`conv2d needs a positive number of filters, got ${filters}`)
		if (inShape.h < 3 || inShape.w < 3) throw new Error(`input ${fullShape(inShape)} is smaller than the 3×3 kernel`)
		this.inShape = inShape
		this.outShape = { c: filters, h: inShape.h - 2, w: inShape.w - 2 }
		this.activation = activation
		this.W = new Float32Array(filters * inShape.c * 9)
		this.b = new Float32Array(filters)
		this.params = [this.W, this.b]
	}

	init(rand: Rand): void {
		heInit(this.W, this.inShape.c * 9, rand)
	}

	/** Offsets of the 9 kernel cells relative to the window's top-left input element. */
	private offsets(): number[] {
		const iw = this.inShape.w
		return [0, 1, 2, iw, iw + 1, iw + 2, 2 * iw, 2 * iw + 1, 2 * iw + 2]
	}

	forward(x: Float32Array, batch: number): LayerForward {
		const { c: C, h: H, w: IW } = this.inShape
		const { c: F, h: OH, w: OW } = this.outShape
		const inSize = C * H * IW
		const outSize = F * OH * OW
		const W = this.W
		const off = this.offsets()
		const pre = new Float32Array(batch * outSize)
		for (let n = 0; n < batch; n++) {
			for (let f = 0; f < F; f++) {
				const bias = this.b[f]
				const po = n * outSize + f * OH * OW
				for (let oy = 0; oy < OH; oy++) {
					for (let ox = 0; ox < OW; ox++) {
						let s = bias
						for (let c = 0; c < C; c++) {
							const xc = n * inSize + c * H * IW + oy * IW + ox
							const wc = (f * C + c) * 9
							for (let k = 0; k < 9; k++) s += x[xc + off[k]] * W[wc + k]
						}
						pre[po + oy * OW + ox] = s
					}
				}
			}
		}
		return { pre }
	}

	backward(
		x: Float32Array,
		dPre: Float32Array,
		batch: number,
		_fwd: LayerForward,
		needInputGrad: boolean,
	): LayerBackward {
		const { c: C, h: H, w: IW } = this.inShape
		const { c: F, h: OH, w: OW } = this.outShape
		const inSize = C * H * IW
		const outSize = F * OH * OW
		const W = this.W
		const off = this.offsets()
		const dW = new Float32Array(W.length)
		const db = new Float32Array(F)
		const dIn = new Float32Array(needInputGrad ? batch * inSize : 0)
		for (let n = 0; n < batch; n++) {
			for (let f = 0; f < F; f++) {
				const po = n * outSize + f * OH * OW
				for (let oy = 0; oy < OH; oy++) {
					for (let ox = 0; ox < OW; ox++) {
						const g = dPre[po + oy * OW + ox]
						if (g === 0) continue
						db[f] += g
						for (let c = 0; c < C; c++) {
							const xc = n * inSize + c * H * IW + oy * IW + ox
							const wc = (f * C + c) * 9
							// Weight sharing: every position this filter visited adds to the same 9 weights.
							for (let k = 0; k < 9; k++) dW[wc + k] += g * x[xc + off[k]]
							if (needInputGrad) for (let k = 0; k < 9; k++) dIn[xc + off[k]] += g * W[wc + k]
						}
					}
				}
			}
		}
		return { dIn, grads: [dW, db] }
	}
}

/** 2×2 max-pooling, stride 2; odd sizes drop the last row / column. */
export class MaxPool implements Layer {
	readonly kind = "maxpool" as const
	readonly inShape: Shape
	readonly outShape: Shape
	readonly activation: Activation = "none"
	readonly params: Float32Array[] = []

	constructor(inShape: Shape) {
		const h = Math.floor(inShape.h / 2)
		const w = Math.floor(inShape.w / 2)
		if (h < 1 || w < 1) throw new Error(`input ${fullShape(inShape)} is too small to pool 2×2`)
		this.inShape = inShape
		this.outShape = { c: inShape.c, h, w }
	}

	init(): void {}

	forward(x: Float32Array, batch: number): LayerForward {
		const { c: C, h: H, w: IW } = this.inShape
		const { h: OH, w: OW } = this.outShape
		const inSize = C * H * IW
		const outSize = C * OH * OW
		const pre = new Float32Array(batch * outSize)
		const argmax = new Int32Array(batch * outSize)
		for (let n = 0; n < batch; n++) {
			for (let c = 0; c < C; c++) {
				for (let oy = 0; oy < OH; oy++) {
					for (let ox = 0; ox < OW; ox++) {
						const i0 = n * inSize + c * H * IW + 2 * oy * IW + 2 * ox
						let best = i0
						if (x[i0 + 1] > x[best]) best = i0 + 1
						if (x[i0 + IW] > x[best]) best = i0 + IW
						if (x[i0 + IW + 1] > x[best]) best = i0 + IW + 1
						const o = n * outSize + (c * OH + oy) * OW + ox
						pre[o] = x[best]
						argmax[o] = best
					}
				}
			}
		}
		return { pre, argmax }
	}

	backward(
		_x: Float32Array,
		dPre: Float32Array,
		batch: number,
		fwd: LayerForward,
		needInputGrad: boolean,
	): LayerBackward {
		const dIn = new Float32Array(needInputGrad ? batch * shapeSize(this.inShape) : 0)
		if (needInputGrad && fwd.argmax) {
			// Only the input that won the max influenced the output, so only it receives the error.
			for (let o = 0; o < dPre.length; o++) dIn[fwd.argmax[o]] += dPre[o]
		}
		return { dIn, grads: [] }
	}
}
