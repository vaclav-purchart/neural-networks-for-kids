import { describe, expect, it } from "vitest"
import { mulberry32 } from "../../src/nn/rng"
import {
	paramShapes,
	Transformer,
	type TransformerConfig,
	type TransformerForward,
} from "../../src/nn/transformer/model"
import {
	ARC_MIN,
	buildTextViewModel,
	caption,
	captionParts,
	mlpWeightsOf,
	SHOWN_DIMS,
	SHOWN_HIDDEN,
	VISIBLE_LETTERS,
} from "../../src/web/text-view-model"

const config: TransformerConfig = { vocab: 6, context: 16, dModel: 32, heads: 2, blocks: 2 }
const labels = ["↵", "␣", "a", "b", "c", "d"]

function run(ids: number[]): { model: Transformer; fwd: TransformerForward } {
	const model = new Transformer(config)
	model.init(mulberry32(3))
	// Larger weights than the init so attention is not uniform.
	const rand = mulberry32(4)
	for (const p of model.params()) for (let i = 0; i < p.length; i++) p[i] += (rand() - 0.5) * 0.8
	return { model, fwd: model.forward(ids, 1, ids.length) }
}

const ids = [2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5] // 14 characters

describe("buildTextViewModel", () => {
	it("shows the last 10 characters and selects the newest by default", () => {
		const { fwd } = run(ids)
		const vm = buildTextViewModel(fwd, config, labels, ids.length - 1, null)
		expect(vm.first).toBe(4)
		expect(vm.letters).toEqual(ids.slice(4).map((i) => labels[i]))
		expect(vm.letters).toHaveLength(VISIBLE_LETTERS)
		expect(vm.selected).toBe(VISIBLE_LETTERS - 1)
		expect(vm.isNewest).toBe(true)
	})

	it("draws the first 16 of the 64 numbers of every visible letter", () => {
		const { fwd } = run(ids)
		const vm = buildTextViewModel(fwd, config, labels, ids.length - 1, null)
		const d = config.dModel
		expect(vm.embedding).toHaveLength(VISIBLE_LETTERS)
		expect(vm.embedding[0]).toHaveLength(SHOWN_DIMS)
		expect(vm.embedding[2][5]).toBeCloseTo(fwd.streams[0][(vm.first + 2) * d + 5])
		expect(vm.blocks[1].output[3][7]).toBeCloseTo(fwd.streams[2][(vm.first + 3) * d + 7])
		const sel = ids.length - 1
		expect(vm.blocks[0].afterAttention[4]).toBeCloseTo(fwd.afterAttention[0][sel * d + 4])
	})

	it("draws the 24 most active hidden MLP neurons of the selected letter, in neuron order", () => {
		const { fwd } = run(ids)
		const sel = ids.length - 1
		const vm = buildTextViewModel(fwd, config, labels, sel, null)
		const F = 4 * config.dModel
		const row = Array.from(fwd.mlpHidden[0].subarray(sel * F, (sel + 1) * F))
		const threshold = [...row].map(Math.abs).sort((a, b) => b - a)[SHOWN_HIDDEN - 1]
		const hidden = vm.blocks[0].hidden
		expect(hidden).toHaveLength(SHOWN_HIDDEN)
		for (const h of hidden) {
			expect(h.value).toBeCloseTo(row[h.index])
			expect(Math.abs(h.value)).toBeGreaterThanOrEqual(threshold - 1e-7)
		}
		for (let i = 1; i < hidden.length; i++) expect(hidden[i].index).toBeGreaterThan(hidden[i - 1].index)
	})

	it("turns attention into arcs: visible letters, one 'earlier text' arc per head, small ones dropped", () => {
		const { fwd } = run(ids)
		const sel = ids.length - 1
		const vm = buildTextViewModel(fwd, config, labels, sel, null)
		const T = ids.length
		for (let h = 0; h < config.heads; h++) {
			const row = fwd.attention[0].subarray((h * T + sel) * T, (h * T + sel + 1) * T)
			const earlier = row.slice(0, vm.first).reduce((a, b) => a + b, 0)
			const arcs = vm.blocks[0].arcs.filter((a) => a.head === h)
			const earlierArc = arcs.find((a) => a.from === "earlier")
			if (earlier >= ARC_MIN) expect(earlierArc?.weight).toBeCloseTo(earlier)
			else expect(earlierArc).toBeUndefined()
			for (const a of arcs) {
				expect(a.weight).toBeGreaterThanOrEqual(ARC_MIN)
				if (a.from !== "earlier") expect(a.weight).toBeCloseTo(row[vm.first + a.from])
			}
		}
		const weights = vm.blocks[0].arcs.map((a) => a.weight)
		expect([...weights].sort((a, b) => b - a)).toEqual(weights)
	})

	it("can show one head only", () => {
		const { fwd } = run(ids)
		const vm = buildTextViewModel(fwd, config, labels, ids.length - 1, 1)
		for (const b of vm.blocks) for (const a of b.arcs) expect(a.head).toBe(1)
	})

	it("predicts the top 5 next characters for the selected position", () => {
		const { fwd } = run(ids)
		const sel = 8
		const vm = buildTextViewModel(fwd, config, labels, sel, null)
		expect(vm.isNewest).toBe(false)
		expect(vm.selected).toBe(sel - vm.first)
		expect(vm.top).toHaveLength(5)
		expect(vm.top[0].p).toBeCloseTo(Math.max(...fwd.probs.subarray(sel * 6, sel * 6 + 6)))
		expect(vm.top[0].char).toBe(labels[vm.top[0].id])
	})

	it("works with fewer characters than the visible window", () => {
		const { fwd } = run([2, 3])
		const vm = buildTextViewModel(fwd, config, labels, 1, null)
		expect(vm.first).toBe(0)
		expect(vm.letters).toEqual(["a", "b"])
		expect(vm.blocks[0].arcs.every((a) => a.from !== "earlier")).toBe(true)
	})

	it("clamps a selection outside the visible letters to the first visible one", () => {
		const { fwd } = run(ids)
		const vm = buildTextViewModel(fwd, config, labels, 0, null)
		expect(vm.selected).toBe(0)
	})
})

describe("MLP connection flows", () => {
	it("are input × weight into each drawn hidden neuron, and hidden × weight out of it", () => {
		const { model, fwd } = run(ids)
		const sel = ids.length - 1
		const vm = buildTextViewModel(fwd, config, labels, sel, null, mlpWeightsOf(model))
		const d = config.dModel
		const F = 4 * d
		const names = paramShapes(config).map((p) => p.name)
		const W1 = model.params()[names.indexOf("b0.W1")]
		const W2 = model.params()[names.indexOf("b0.W2")]
		const { hidden, inFlows, outFlows } = vm.blocks[0]
		expect(inFlows).toHaveLength(SHOWN_DIMS)
		expect(inFlows[0]).toHaveLength(SHOWN_HIDDEN)
		expect(outFlows).toHaveLength(SHOWN_HIDDEN)
		expect(outFlows[0]).toHaveLength(SHOWN_DIMS)
		const j = 5
		const k = 3
		expect(inFlows[k][j]).toBeCloseTo(fwd.mlpInput[0][sel * d + k] * W1[hidden[j].index * d + k], 6)
		expect(outFlows[j][k]).toBeCloseTo(hidden[j].value * W2[k * F + hidden[j].index], 6)
	})

	it("are empty without weights", () => {
		const { fwd } = run(ids)
		const vm = buildTextViewModel(fwd, config, labels, ids.length - 1, null)
		expect(vm.blocks[0].inFlows).toEqual([])
		expect(vm.blocks[0].outFlows).toEqual([])
	})
})

describe("captions", () => {
	const { fwd } = run(ids)
	const vm = buildTextViewModel(fwd, config, labels, ids.length - 1, null)

	it("explain every step with the real values", () => {
		expect(caption(0, vm, null, 0.8)).toContain(`"${vm.letters[vm.selected]}"`)
		expect(caption(1, vm, null, 0.8)).toContain("32 numbers")
		const strongest = vm.blocks[0].arcs[0]
		expect(caption(2, vm, null, 0.8)).toContain(`${Math.round(strongest.weight * 100)}%`)
		expect(caption(3, vm, null, 0.8)).toContain("32 → 128 → 32")
		expect(caption(5, vm, null, 0.8)).toContain(`${Math.round(vm.top[0].p * 100)}%`)
		const pick = { char: "c", p: 0.42 }
		const pickText = caption(6, vm, pick, 0.8)
		expect(pickText).toContain('"c"')
		expect(pickText).toContain("42%")
		expect(pickText).toContain("0.80")
	})

	it("explain that inner positions predict too, and nothing is added for them", () => {
		const inner = buildTextViewModel(fwd, config, labels, 8, null)
		expect(caption(5, inner, null, 0.8)).toMatch(/every position/i)
		expect(caption(6, inner, { char: "a", p: 0.3 }, 0.8)).toMatch(/nothing is added/i)
	})

	it("split **bold** markers into parts", () => {
		expect(captionParts("a **b** c")).toEqual([
			{ text: "a ", bold: false },
			{ text: "b", bold: true },
			{ text: " c", bold: false },
		])
	})
})
