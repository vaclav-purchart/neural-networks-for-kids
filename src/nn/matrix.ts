/**
 * Dense-layer kernels on row-major Float32Arrays.
 * Shapes: x [batch × inDim], W [outDim × inDim], b [outDim], z [batch × outDim].
 * Inner loops always walk contiguous memory.
 */

/** z = x · Wᵀ + b */
export function linearForward(
	x: Float32Array,
	W: Float32Array,
	b: Float32Array,
	z: Float32Array,
	batch: number,
	inDim: number,
	outDim: number,
): void {
	for (let n = 0; n < batch; n++) {
		const xo = n * inDim
		const zo = n * outDim
		for (let o = 0; o < outDim; o++) {
			const wo = o * inDim
			let s = b[o]
			for (let i = 0; i < inDim; i++) s += x[xo + i] * W[wo + i]
			z[zo + o] = s
		}
	}
}

/** dW = dzᵀ · x and db = column sums of dz (both overwritten). */
export function linearWeightGrad(
	dz: Float32Array,
	x: Float32Array,
	dW: Float32Array,
	db: Float32Array,
	batch: number,
	inDim: number,
	outDim: number,
): void {
	dW.fill(0)
	db.fill(0)
	for (let n = 0; n < batch; n++) {
		const xo = n * inDim
		const zo = n * outDim
		for (let o = 0; o < outDim; o++) {
			const g = dz[zo + o]
			if (g === 0) continue
			db[o] += g
			const wo = o * inDim
			for (let i = 0; i < inDim; i++) dW[wo + i] += g * x[xo + i]
		}
	}
}

/** dx = dz · W (overwritten). */
export function linearInputGrad(
	dz: Float32Array,
	W: Float32Array,
	dx: Float32Array,
	batch: number,
	inDim: number,
	outDim: number,
): void {
	dx.fill(0)
	for (let n = 0; n < batch; n++) {
		const xo = n * inDim
		const zo = n * outDim
		for (let o = 0; o < outDim; o++) {
			const g = dz[zo + o]
			if (g === 0) continue
			const wo = o * inDim
			for (let i = 0; i < inDim; i++) dx[xo + i] += g * W[wo + i]
		}
	}
}
