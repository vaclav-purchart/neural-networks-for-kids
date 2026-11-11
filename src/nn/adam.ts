export interface AdamOptions {
	lr?: number
	beta1?: number
	beta2?: number
	eps?: number
}

/** Adam optimiser: per-parameter adaptive learning rates from gradient moment estimates. */
export class Adam {
	/** Learning rate; the trainer may lower it during training. */
	lr: number
	private readonly beta1: number
	private readonly beta2: number
	private readonly eps: number
	private readonly params: readonly Float32Array[]
	private readonly m: Float32Array[]
	private readonly v: Float32Array[]
	private t = 0

	/** params: the tensors to optimise (e.g. Network.params()); grads passed to step() use the same order. */
	constructor(params: readonly Float32Array[], opts: AdamOptions = {}) {
		this.params = params
		this.lr = opts.lr ?? 1e-3
		this.beta1 = opts.beta1 ?? 0.9
		this.beta2 = opts.beta2 ?? 0.999
		this.eps = opts.eps ?? 1e-8
		this.m = params.map((p) => new Float32Array(p.length))
		this.v = params.map((p) => new Float32Array(p.length))
	}

	step(grads: readonly Float32Array[]): void {
		this.t++
		const c1 = 1 - this.beta1 ** this.t
		const c2 = 1 - this.beta2 ** this.t
		const { beta1, beta2, lr, eps } = this
		this.params.forEach((p, slot) => {
			const g = grads[slot]
			const m = this.m[slot]
			const v = this.v[slot]
			for (let i = 0; i < p.length; i++) {
				m[i] = beta1 * m[i] + (1 - beta1) * g[i]
				v[i] = beta2 * v[i] + (1 - beta2) * g[i] * g[i]
				p[i] -= (lr * (m[i] / c1)) / (Math.sqrt(v[i] / c2) + eps)
			}
		})
	}
}
