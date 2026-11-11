import { describe, expect, it } from "vitest"
import { mulberry32 } from "../../../src/nn/rng"
import { inspectPosition } from "../../../src/nn/transformer/inspect"
import { Transformer } from "../../../src/nn/transformer/model"

describe("inspectPosition", () => {
	it("lists each head's strongest earlier positions and the top predictions", () => {
		const config = { vocab: 6, context: 5, dModel: 8, heads: 2, blocks: 2 }
		const model = new Transformer(config)
		model.init(mulberry32(4))
		const fwd = model.forward([1, 2, 3, 4], 1, 4)
		const d = inspectPosition(fwd, config, 2, 3)
		expect(d.position).toBe(2)
		expect(d.heads).toHaveLength(4) // 2 blocks × 2 heads
		for (const h of d.heads) {
			expect(h.top.length).toBeLessThanOrEqual(3) // only positions 0..2 exist
			for (const a of h.top) expect(a.position).toBeLessThanOrEqual(2)
			for (let i = 1; i < h.top.length; i++) expect(h.top[i - 1].weight).toBeGreaterThanOrEqual(h.top[i].weight)
		}
		expect(d.predictions).toHaveLength(3)
		expect(d.predictions[0].p).toBeGreaterThanOrEqual(d.predictions[1].p)
	})
})
