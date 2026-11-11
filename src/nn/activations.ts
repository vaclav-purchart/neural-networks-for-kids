export function relu(z: Float32Array, out: Float32Array): void {
	for (let i = 0; i < z.length; i++) out[i] = z[i] > 0 ? z[i] : 0
}

/** Row-wise softmax; subtracts the row max so large logits don't overflow. */
export function softmaxRows(z: Float32Array, out: Float32Array, rows: number, cols: number): void {
	for (let r = 0; r < rows; r++) {
		const o = r * cols
		let max = -Infinity
		for (let c = 0; c < cols; c++) if (z[o + c] > max) max = z[o + c]
		let sum = 0
		for (let c = 0; c < cols; c++) {
			const e = Math.exp(z[o + c] - max)
			out[o + c] = e
			sum += e
		}
		for (let c = 0; c < cols; c++) out[o + c] /= sum
	}
}

/** Mean cross-entropy of softmax probabilities against integer class labels. */
export function crossEntropy(probs: Float32Array, labels: ArrayLike<number>, rows: number, cols: number): number {
	let total = 0
	for (let r = 0; r < rows; r++) total -= Math.log(Math.max(probs[r * cols + labels[r]], 1e-12))
	return total / rows
}
