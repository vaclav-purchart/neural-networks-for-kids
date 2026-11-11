/** Transpose every rows×cols image in place. */
export function transposeImages(images: Uint8Array, rows: number, cols: number): void {
	if (rows !== cols) throw new Error("transposeImages expects square images")
	const size = rows * cols
	for (let o = 0; o < images.length; o += size) {
		for (let r = 0; r < rows; r++) {
			for (let c = r + 1; c < cols; c++) {
				const a = o + r * cols + c
				const b = o + c * cols + r
				const t = images[a]
				images[a] = images[b]
				images[b] = t
			}
		}
	}
}

/** Apply a label offset and check every label is a valid class index. */
export function remapLabels(labels: Uint8Array, offset: number, numClasses: number): Uint8Array {
	const out = new Uint8Array(labels.length)
	for (let i = 0; i < labels.length; i++) {
		const v = labels[i] + offset
		if (v < 0 || v >= numClasses) {
			throw new Error(`Label ${labels[i]} at index ${i} maps to ${v}, outside 0..${numClasses - 1}`)
		}
		out[i] = v
	}
	return out
}
