import { crossEntropy, relu, softmaxRows } from "./activations"
import type { Layer, LayerForward } from "./layers"
import type { Rand } from "./rng"
import { describeShape, type Shape, shapeSize } from "./tensor-shape"

export interface ForwardResult {
	batch: number
	/** pre[i]: output of layer i before its activation. */
	pre: Float32Array[]
	/** act[0]: the input; act[i+1]: output of layer i after its activation (softmax probabilities last). */
	act: Float32Array[]
	/** Per-layer data the backward pass needs (e.g. max-pool choices). */
	forwards: LayerForward[]
}

/** A stack of layers ending in a softmax classifier. */
export class Network {
	readonly inputShape: Shape
	readonly layers: readonly Layer[]
	readonly inputSize: number
	readonly classes: number

	constructor(inputShape: Shape, layers: readonly Layer[]) {
		if (layers.length === 0) throw new Error("A network needs at least one layer")
		let shape = inputShape
		layers.forEach((layer, i) => {
			if (shapeSize(layer.inShape) !== shapeSize(shape)) {
				throw new Error(`layer ${i} expects ${describeShape(layer.inShape)} but receives ${describeShape(shape)}`)
			}
			shape = layer.outShape
		})
		const last = layers[layers.length - 1]
		if (last.kind !== "dense" || last.activation !== "softmax")
			throw new Error("The last layer must be dense with softmax")
		this.inputShape = inputShape
		this.layers = layers
		this.inputSize = shapeSize(inputShape)
		this.classes = shapeSize(last.outShape)
	}

	init(rand: Rand): void {
		for (const layer of this.layers) layer.init(rand)
	}

	/** Every learnable tensor, layer by layer (W then b). Adam and snapshot files use this order. */
	params(): Float32Array[] {
		return this.layers.flatMap((l) => l.params)
	}

	setParams(values: readonly Float32Array[]): void {
		const params = this.params()
		if (values.length !== params.length) throw new Error(`Expected ${params.length} tensors, got ${values.length}`)
		values.forEach((v, i) => {
			if (v.length !== params[i].length) {
				throw new Error(`Tensor ${i} has wrong shape (${v.length} values, expected ${params[i].length})`)
			}
		})
		values.forEach((v, i) => {
			params[i].set(v)
		})
	}

	forward(x: Float32Array, batch: number): ForwardResult {
		if (x.length !== batch * this.inputSize) throw new Error(`Input length ${x.length} != ${batch} × ${this.inputSize}`)
		const pre: Float32Array[] = []
		const act: Float32Array[] = [x]
		const forwards: LayerForward[] = []
		for (const layer of this.layers) {
			const fwd = layer.forward(act[act.length - 1], batch)
			const out = new Float32Array(fwd.pre.length)
			if (layer.activation === "softmax") softmaxRows(fwd.pre, out, batch, shapeSize(layer.outShape))
			else if (layer.activation === "relu") relu(fwd.pre, out)
			else out.set(fwd.pre)
			pre.push(fwd.pre)
			act.push(out)
			forwards.push(fwd)
		}
		return { batch, pre, act, forwards }
	}

	/** Class probabilities for a single input. */
	predict(x: Float32Array): Float32Array {
		const result = this.forward(x, 1)
		return result.act[result.act.length - 1]
	}

	loss(result: ForwardResult, labels: ArrayLike<number>): number {
		return crossEntropy(result.act[result.act.length - 1], labels, result.batch, this.classes)
	}

	/** Gradients of the mean cross-entropy loss, aligned with params(). */
	backward(result: ForwardResult, labels: ArrayLike<number>): Float32Array[] {
		const { batch, pre, act, forwards } = result
		const L = this.layers.length
		// Softmax + cross-entropy combine to a simple gradient: (p - onehot) / batch.
		let dPre: Float32Array = new Float32Array(act[L])
		for (let n = 0; n < batch; n++) dPre[n * this.classes + labels[n]] -= 1
		for (let i = 0; i < dPre.length; i++) dPre[i] /= batch

		const grads: Float32Array[][] = new Array(L)
		for (let i = L - 1; i >= 0; i--) {
			const { dIn, grads: g } = this.layers[i].backward(act[i], dPre, batch, forwards[i], i > 0)
			grads[i] = g
			if (i > 0) {
				if (this.layers[i - 1].activation === "relu") {
					const z = pre[i - 1]
					for (let j = 0; j < dIn.length; j++) if (z[j] <= 0) dIn[j] = 0 // ReLU derivative
				}
				dPre = dIn
			}
		}
		return grads.flat()
	}
}
