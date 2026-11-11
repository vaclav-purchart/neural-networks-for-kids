import { describe, expect, it } from "vitest"
import { Conv2d, Dense, MaxPool } from "../../src/nn/layers"
import { vectorShape } from "../../src/nn/tensor-shape"

describe("Conv2d", () => {
	it("slides a 3×3 filter over the input", () => {
		const conv = new Conv2d({ c: 1, h: 3, w: 4 }, 1, "relu")
		conv.W.set([1, 0, 0, 0, 1, 0, 0, 0, 1]) // diagonal
		conv.b[0] = 0.5
		const x = Float32Array.from({ length: 12 }, (_, i) => i) // rows [0..3], [4..7], [8..11]
		expect(conv.outShape).toEqual({ c: 1, h: 1, w: 2 })
		expect([...conv.forward(x, 1).pre]).toEqual([15.5, 18.5]) // 0+5+10+0.5, 1+6+11+0.5
	})

	it("sums over input channels", () => {
		const conv = new Conv2d({ c: 2, h: 3, w: 3 }, 1, "relu")
		conv.W.set([...Array(9).fill(1), ...Array(9).fill(-1)])
		const x = Float32Array.from([...Array(9).fill(1), ...Array(9).fill(2)])
		expect([...conv.forward(x, 1).pre]).toEqual([-9])
	})

	it("rejects inputs smaller than the kernel", () => {
		expect(() => new Conv2d({ c: 1, h: 2, w: 5 }, 1, "relu")).toThrow(/smaller than the 3×3 kernel/)
	})
})

describe("MaxPool", () => {
	// 1×3×5: rows [1,5,2,0,9], [3,4,8,1,9], [9,9,9,9,9]
	const x = Float32Array.from([1, 5, 2, 0, 9, 3, 4, 8, 1, 9, 9, 9, 9, 9, 9])
	const pool = new MaxPool({ c: 1, h: 3, w: 5 })

	it("keeps the max of each 2×2 window and drops the odd last row/column", () => {
		expect(pool.outShape).toEqual({ c: 1, h: 1, w: 2 })
		expect([...pool.forward(x, 1).pre]).toEqual([5, 8])
	})

	it("routes the error only to the chosen input", () => {
		const { dIn, grads } = pool.backward(x, Float32Array.from([10, 20]), 1, pool.forward(x, 1), true)
		const expected = new Array(15).fill(0)
		expected[1] = 10
		expected[7] = 20
		expect([...dIn]).toEqual(expected)
		expect(grads).toEqual([])
	})

	it("sends the error to a single input when a window is a tie", () => {
		const flat = new Float32Array(4).fill(0.5)
		const p = new MaxPool({ c: 1, h: 2, w: 2 })
		const { dIn } = p.backward(flat, Float32Array.from([1]), 1, p.forward(flat, 1), true)
		expect([...dIn]).toEqual([1, 0, 0, 0])
	})

	it("rejects inputs too small to pool", () => {
		expect(() => new MaxPool({ c: 1, h: 1, w: 4 })).toThrow(/too small to pool/)
	})
})

describe("Dense", () => {
	it("computes W·x + b", () => {
		const d = new Dense(vectorShape(2), 2, "relu")
		d.W.set([1, 2, 3, 4])
		d.b.set([0.5, -1])
		expect([...d.forward(Float32Array.from([1, 1]), 1).pre]).toEqual([3.5, 6])
	})
})
