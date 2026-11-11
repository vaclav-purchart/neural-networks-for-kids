import { describe, expect, it } from "vitest"
import { parseIdx } from "../../src/data/idx"

function makeIdx(shape: number[], data: number[], typeByte = 0x08): Uint8Array {
	const header = new Uint8Array(4 + 4 * shape.length)
	header[2] = typeByte
	header[3] = shape.length
	const view = new DataView(header.buffer)
	shape.forEach((d, i) => {
		view.setUint32(4 + 4 * i, d, false)
	})
	return Uint8Array.from([...header, ...data])
}

describe("parseIdx", () => {
	it("parses shape and data", () => {
		const r = parseIdx(makeIdx([2, 2, 1], [1, 2, 3, 4]))
		expect(r.shape).toEqual([2, 2, 1])
		expect([...r.data]).toEqual([1, 2, 3, 4])
	})

	it("rejects bad magic, wrong type, truncated header and wrong length", () => {
		const good = makeIdx([2], [1, 2])
		const badMagic = good.slice()
		badMagic[0] = 1
		expect(() => parseIdx(badMagic, "x.idx")).toThrow(/x\.idx: bad magic/)
		expect(() => parseIdx(makeIdx([2], [1, 2], 0x0d))).toThrow(/unsupported data type/)
		expect(() => parseIdx(good.subarray(0, 6))).toThrow(/truncated header/)
		expect(() => parseIdx(makeIdx([3], [1, 2]))).toThrow(/expected 3 bytes/)
	})
})
