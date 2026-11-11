/**
 * Transformer building blocks on row-major Float32Arrays. A "row" is one position of one sequence:
 * row r = n·T + t for sequence n and position t. Backward functions overwrite input gradients and
 * accumulate (+=) parameter gradients, so callers zero parameter-gradient buffers once per step.
 */

export const LN_EPS = 1e-5

export function addInto(target: Float32Array, src: Float32Array): void {
	for (let i = 0; i < target.length; i++) target[i] += src[i]
}

/** out[n,t] = tok[ids[n,t]] + pos[t] */
export function embeddingForward(
	ids: ArrayLike<number>,
	tok: Float32Array,
	pos: Float32Array,
	out: Float32Array,
	batch: number,
	T: number,
	d: number,
): void {
	for (let n = 0; n < batch; n++) {
		for (let t = 0; t < T; t++) {
			const r = (n * T + t) * d
			const to = ids[n * T + t] * d
			const po = t * d
			for (let i = 0; i < d; i++) out[r + i] = tok[to + i] + pos[po + i]
		}
	}
}

export function embeddingBackward(
	ids: ArrayLike<number>,
	dOut: Float32Array,
	dTok: Float32Array,
	dPos: Float32Array,
	batch: number,
	T: number,
	d: number,
): void {
	for (let n = 0; n < batch; n++) {
		for (let t = 0; t < T; t++) {
			const r = (n * T + t) * d
			const to = ids[n * T + t] * d
			const po = t * d
			for (let i = 0; i < d; i++) {
				dTok[to + i] += dOut[r + i]
				dPos[po + i] += dOut[r + i]
			}
		}
	}
}

export interface LayerNormCache {
	/** Normalized input (before gain and bias). */
	xhat: Float32Array
	/** 1 / √(variance + ε) per row. */
	rstd: Float32Array
}

/** y = g ⊙ (x − mean) / √(var + ε) + b, per row of length d. */
export function layerNormForward(
	x: Float32Array,
	g: Float32Array,
	b: Float32Array,
	y: Float32Array,
	rows: number,
	d: number,
): LayerNormCache {
	const xhat = new Float32Array(rows * d)
	const rstd = new Float32Array(rows)
	for (let r = 0; r < rows; r++) {
		const o = r * d
		let mean = 0
		for (let i = 0; i < d; i++) mean += x[o + i]
		mean /= d
		let variance = 0
		for (let i = 0; i < d; i++) variance += (x[o + i] - mean) ** 2
		variance /= d
		const rs = 1 / Math.sqrt(variance + LN_EPS)
		rstd[r] = rs
		for (let i = 0; i < d; i++) {
			const xh = (x[o + i] - mean) * rs
			xhat[o + i] = xh
			y[o + i] = g[i] * xh + b[i]
		}
	}
	return { xhat, rstd }
}

export function layerNormBackward(
	dy: Float32Array,
	cache: LayerNormCache,
	g: Float32Array,
	dx: Float32Array,
	dg: Float32Array,
	db: Float32Array,
	rows: number,
	d: number,
): void {
	const { xhat, rstd } = cache
	for (let r = 0; r < rows; r++) {
		const o = r * d
		let sum = 0
		let sumXhat = 0
		for (let i = 0; i < d; i++) {
			const dxh = dy[o + i] * g[i]
			sum += dxh
			sumXhat += dxh * xhat[o + i]
			dg[i] += dy[o + i] * xhat[o + i]
			db[i] += dy[o + i]
		}
		const k = rstd[r] / d
		for (let i = 0; i < d; i++) {
			const dxh = dy[o + i] * g[i]
			dx[o + i] = k * (d * dxh - sum - xhat[o + i] * sumXhat)
		}
	}
}

const GELU_C = Math.sqrt(2 / Math.PI)

/** GELU, tanh approximation (as in GPT-2). */
export function gelu(x: Float32Array, y: Float32Array): void {
	for (let i = 0; i < x.length; i++) {
		const v = x[i]
		y[i] = 0.5 * v * (1 + Math.tanh(GELU_C * (v + 0.044715 * v * v * v)))
	}
}

export function geluBackward(x: Float32Array, dy: Float32Array, dx: Float32Array): void {
	for (let i = 0; i < x.length; i++) {
		const v = x[i]
		const th = Math.tanh(GELU_C * (v + 0.044715 * v * v * v))
		const du = GELU_C * (1 + 3 * 0.044715 * v * v)
		dx[i] = dy[i] * (0.5 * (1 + th) + 0.5 * v * (1 - th * th) * du)
	}
}

/**
 * Causal multi-head self-attention. q, k, v, out: [batch·T × d]; head h uses columns h·hs … (h+1)·hs − 1
 * (hs = d / H). att: [batch × H × T × T], att[n,h,t,s] = softmax over s ≤ t of q_t·k_s / √hs (0 for s > t).
 */
export function attentionForward(
	q: Float32Array,
	k: Float32Array,
	v: Float32Array,
	att: Float32Array,
	out: Float32Array,
	batch: number,
	T: number,
	d: number,
	H: number,
): void {
	const hs = d / H
	const scale = 1 / Math.sqrt(hs)
	for (let n = 0; n < batch; n++) {
		for (let h = 0; h < H; h++) {
			const col = h * hs
			for (let t = 0; t < T; t++) {
				const row = ((n * H + h) * T + t) * T
				const qo = (n * T + t) * d + col
				let max = -Infinity
				for (let s = 0; s <= t; s++) {
					const ko = (n * T + s) * d + col
					let score = 0
					for (let j = 0; j < hs; j++) score += q[qo + j] * k[ko + j]
					score *= scale
					att[row + s] = score
					if (score > max) max = score
				}
				let sum = 0
				for (let s = 0; s <= t; s++) {
					const e = Math.exp(att[row + s] - max)
					att[row + s] = e
					sum += e
				}
				for (let s = 0; s <= t; s++) att[row + s] /= sum
				for (let s = t + 1; s < T; s++) att[row + s] = 0
				const oo = (n * T + t) * d + col
				for (let j = 0; j < hs; j++) out[oo + j] = 0
				for (let s = 0; s <= t; s++) {
					const w = att[row + s]
					const vo = (n * T + s) * d + col
					for (let j = 0; j < hs; j++) out[oo + j] += w * v[vo + j]
				}
			}
		}
	}
}

export function attentionBackward(
	q: Float32Array,
	k: Float32Array,
	v: Float32Array,
	att: Float32Array,
	dOut: Float32Array,
	dq: Float32Array,
	dk: Float32Array,
	dv: Float32Array,
	batch: number,
	T: number,
	d: number,
	H: number,
): void {
	const hs = d / H
	const scale = 1 / Math.sqrt(hs)
	dq.fill(0)
	dk.fill(0)
	dv.fill(0)
	const dA = new Float32Array(T)
	for (let n = 0; n < batch; n++) {
		for (let h = 0; h < H; h++) {
			const col = h * hs
			for (let t = 0; t < T; t++) {
				const row = ((n * H + h) * T + t) * T
				const to = (n * T + t) * d + col
				// out_t = Σ_s att[t,s] · v_s
				for (let s = 0; s <= t; s++) {
					const so = (n * T + s) * d + col
					const w = att[row + s]
					let g = 0
					for (let j = 0; j < hs; j++) {
						g += dOut[to + j] * v[so + j]
						dv[so + j] += w * dOut[to + j]
					}
					dA[s] = g
				}
				// softmax backward: dScore_s = att_s · (dA_s − Σ att · dA)
				let weighted = 0
				for (let s = 0; s <= t; s++) weighted += att[row + s] * dA[s]
				for (let s = 0; s <= t; s++) {
					const dScore = att[row + s] * (dA[s] - weighted) * scale
					if (dScore === 0) continue
					const so = (n * T + s) * d + col
					for (let j = 0; j < hs; j++) {
						dq[to + j] += dScore * k[so + j]
						dk[so + j] += dScore * q[to + j]
					}
				}
			}
		}
	}
}
