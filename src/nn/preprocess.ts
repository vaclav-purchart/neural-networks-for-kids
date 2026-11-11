/**
 * MNIST digits are centred by centre of mass inside the 28×28 field. Doing the
 * same to a hand drawing makes predictions far less sensitive to where you draw.
 * Pixels are 0..1. A blank image comes back as a blank copy.
 */
export function centerByMass(pixels: ArrayLike<number>, rows: number, cols: number): Float32Array {
	let mass = 0
	let sumY = 0
	let sumX = 0
	for (let r = 0; r < rows; r++) {
		for (let c = 0; c < cols; c++) {
			const v = pixels[r * cols + c]
			mass += v
			sumY += v * r
			sumX += v * c
		}
	}
	const out = new Float32Array(rows * cols)
	if (mass < 1e-6) {
		out.set(pixels)
		return out
	}
	const dy = Math.round((rows - 1) / 2 - sumY / mass)
	const dx = Math.round((cols - 1) / 2 - sumX / mass)
	for (let r = 0; r < rows; r++) {
		const sr = r - dy
		if (sr < 0 || sr >= rows) continue
		for (let c = 0; c < cols; c++) {
			const sc = c - dx
			if (sc >= 0 && sc < cols) out[r * cols + c] = pixels[sr * cols + sc]
		}
	}
	return out
}

/** (x - mean) / std — must match what the trainer did. */
export function normalize(pixels: ArrayLike<number>, mean: number, std: number): Float32Array {
	const out = new Float32Array(pixels.length)
	for (let i = 0; i < pixels.length; i++) out[i] = (pixels[i] - mean) / std
	return out
}

/** Area-weighted average of the source rectangle [y0, y1) × [x0, x1) (fractional pixel edges allowed). */
function boxAverage(p: ArrayLike<number>, cols: number, y0: number, y1: number, x0: number, x1: number): number {
	let sum = 0
	let area = 0
	for (let y = Math.floor(y0); y < Math.ceil(y1); y++) {
		const wy = Math.min(y + 1, y1) - Math.max(y, y0)
		if (wy <= 0) continue
		for (let x = Math.floor(x0); x < Math.ceil(x1); x++) {
			const wx = Math.min(x + 1, x1) - Math.max(x, x0)
			if (wx <= 0) continue
			sum += p[y * cols + x] * wx * wy
			area += wx * wy
		}
	}
	return area ? sum / area : 0
}

/** Bilinear sample at pixel-centre coordinates (y, x); outside the image counts as 0 (no ink). */
function sampleBilinear(p: ArrayLike<number>, rows: number, cols: number, y: number, x: number): number {
	const y0 = Math.floor(y)
	const x0 = Math.floor(x)
	const fy = y - y0
	const fx = x - x0
	const at = (r: number, c: number) => (r < 0 || r >= rows || c < 0 || c >= cols ? 0 : p[r * cols + c])
	return (
		(1 - fy) * ((1 - fx) * at(y0, x0) + fx * at(y0, x0 + 1)) +
		fy * ((1 - fx) * at(y0 + 1, x0) + fx * at(y0 + 1, x0 + 1))
	)
}

/**
 * Scale the drawing so its ink fits a fitBox × fitBox square (aspect ratio kept), the way the
 * datasets were prepared (MNIST: 20, EMNIST: ~24), and place it in the middle of the field.
 * Pixels are 0..1; a blank drawing, or a mere dot (ink ≤ 2×2 px), comes back unchanged.
 * Follow with centerByMass.
 */
export function normalizeSize(
	pixels: ArrayLike<number>,
	rows: number,
	cols: number,
	fitBox: number,
	threshold = 0.05,
): Float32Array {
	let top = rows
	let bottom = -1
	let left = cols
	let right = -1
	for (let r = 0; r < rows; r++) {
		for (let c = 0; c < cols; c++) {
			if (pixels[r * cols + c] > threshold) {
				top = Math.min(top, r)
				bottom = Math.max(bottom, r)
				left = Math.min(left, c)
				right = Math.max(right, c)
			}
		}
	}
	const out = new Float32Array(rows * cols)
	const bh = bottom - top + 1
	const bw = right - left + 1
	// Nothing drawn, or just a tap: scaling a dot up to a full character would be invented ink.
	if (bottom < 0 || Math.max(bh, bw) <= 2) {
		out.set(pixels)
		return out
	}
	const scale = fitBox / Math.max(bh, bw)
	const nh = Math.min(rows, Math.max(1, Math.round(bh * scale)))
	const nw = Math.min(cols, Math.max(1, Math.round(bw * scale)))
	const sy = bh / nh // source pixels per target pixel
	const sx = bw / nw
	const oy = Math.floor((rows - nh) / 2)
	const ox = Math.floor((cols - nw) / 2)
	for (let ty = 0; ty < nh; ty++) {
		for (let tx = 0; tx < nw; tx++) {
			const v =
				scale > 1
					? sampleBilinear(pixels, rows, cols, top + (ty + 0.5) * sy - 0.5, left + (tx + 0.5) * sx - 0.5)
					: boxAverage(pixels, cols, top + ty * sy, top + (ty + 1) * sy, left + tx * sx, left + (tx + 1) * sx)
			out[(oy + ty) * cols + ox + tx] = Math.min(1, Math.max(0, v))
		}
	}
	return out
}

/** How the dataset centred its characters: MNIST by centre of mass, EMNIST by bounding box. */
export type Centering = "mass" | "box"

export interface Preprocessing {
	/** Ink is scaled to fit fitBox × fitBox. */
	fitBox: number
	center: Centering
}

/** Shift the image so the bounding box of its ink sits in the middle (no scaling). Blank → blank copy. */
export function centerByBox(pixels: ArrayLike<number>, rows: number, cols: number, threshold = 0.05): Float32Array {
	let top = rows
	let bottom = -1
	let left = cols
	let right = -1
	for (let r = 0; r < rows; r++) {
		for (let c = 0; c < cols; c++) {
			if (pixels[r * cols + c] > threshold) {
				top = Math.min(top, r)
				bottom = Math.max(bottom, r)
				left = Math.min(left, c)
				right = Math.max(right, c)
			}
		}
	}
	const out = new Float32Array(rows * cols)
	if (bottom < 0) {
		out.set(pixels)
		return out
	}
	const dy = Math.floor((rows - (bottom - top + 1)) / 2) - top
	const dx = Math.floor((cols - (right - left + 1)) / 2) - left
	for (let r = 0; r < rows; r++) {
		const sr = r - dy
		if (sr < 0 || sr >= rows) continue
		for (let c = 0; c < cols; c++) {
			const sc = c - dx
			if (sc >= 0 && sc < cols) out[r * cols + c] = pixels[sr * cols + sc]
		}
	}
	return out
}

/**
 * What the network sees for a drawing: optionally scaled to the dataset's character size, then
 * centred the way the dataset was centred. Values stay 0..1 (normalise afterwards).
 */
export function prepareDrawing(
	pixels: ArrayLike<number>,
	rows: number,
	cols: number,
	pre: Preprocessing,
	scale: boolean,
): Float32Array {
	const sized = scale ? normalizeSize(pixels, rows, cols, pre.fitBox) : pixels
	return pre.center === "box" ? centerByBox(sized, rows, cols) : centerByMass(sized, rows, cols)
}
