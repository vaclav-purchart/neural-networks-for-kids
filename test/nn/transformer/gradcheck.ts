import { expect } from "vitest"

/** Central-difference check of `grads` (one array per param) against `lossAt()`. */
export function expectGradients(params: Float32Array[], grads: Float32Array[], lossAt: () => number, eps = 1e-3): void {
	params.forEach((param, p) => {
		for (let i = 0; i < param.length; i++) {
			const orig = param[i]
			param[i] = orig + eps
			const plus = lossAt()
			param[i] = orig - eps
			const minus = lossAt()
			param[i] = orig
			const numeric = (plus - minus) / (2 * eps)
			const tolerance = 2e-3 + 2e-2 * Math.max(Math.abs(numeric), Math.abs(grads[p][i]))
			expect(Math.abs(numeric - grads[p][i]), `param ${p}[${i}]`).toBeLessThan(tolerance)
		}
	})
}
