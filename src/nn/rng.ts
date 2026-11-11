/** Seeded random numbers so training runs are reproducible. */
export type Rand = () => number

/** Mulberry32: tiny, fast PRNG returning floats in [0, 1). */
export function mulberry32(seed: number): Rand {
	let a = seed >>> 0
	return () => {
		a = (a + 0x6d2b79f5) >>> 0
		let t = a
		t = Math.imul(t ^ (t >>> 15), t | 1)
		t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296
	}
}

/** Standard normal sample (Box–Muller). */
export function randomNormal(rand: Rand): number {
	let u = 0
	while (u === 0) u = rand()
	return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand())
}

/** Fisher–Yates shuffle. */
export function shuffleInPlace(arr: Uint32Array, rand: Rand): void {
	for (let i = arr.length - 1; i > 0; i--) {
		const j = Math.floor(rand() * (i + 1))
		const t = arr[i]
		arr[i] = arr[j]
		arr[j] = t
	}
}
