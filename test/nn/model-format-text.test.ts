import { describe, expect, it } from "vitest"
import {
	textConfig,
	validateAnyModelInfo,
	validateIndex,
	validateModelInfo,
	validateTextModelInfo,
} from "../../src/nn/model-format"

const textInfo = {
	formatVersion: 2,
	id: "shakespeare",
	arch: "transformer",
	dataset: "shakespeare",
	title: "Tiny Shakespeare",
	labels: ["\n", " ", "a", "b"],
	transformer: { context: 64, dModel: 64, heads: 4, blocks: 2 },
	trainedAt: "2026-10-15T00:00:00.000Z",
	snapshots: [
		{
			file: "snapshots/000.json",
			steps: 0,
			charsSeen: 0,
			testLoss: 4.2,
			testAccuracy: 0.02,
			testSamples: 4096,
			avgLoss: null,
			sample: "ab ba",
		},
	],
}

/** A format v1 image model.json (MLP, before ids and archs existed). */
const v1Info = {
	formatVersion: 1,
	dataset: "mnist",
	title: "MNIST digits",
	labels: ["a", "b"],
	inputShape: [28, 28],
	layerSizes: [784, 3, 2],
	normalization: { mean: 0.1, std: 0.3 },
	trainedAt: "2026-10-12T00:00:00.000Z",
	snapshots: [
		{ file: "snapshots/000.json", imagesSeen: 0, epoch: 0, testAccuracy: 0.5, testSamples: 10, avgLoss: null },
	],
	samples: null,
}

describe("text model.json", () => {
	it("validates and gives the transformer config", () => {
		const info = validateTextModelInfo(textInfo)
		expect(textConfig(info)).toEqual({ vocab: 4, context: 64, dModel: 64, heads: 4, blocks: 2 })
	})

	it("is recognized by validateAnyModelInfo", () => {
		expect(validateAnyModelInfo(textInfo).arch).toBe("transformer")
	})

	it("rejects a bad transformer config", () => {
		const bad = { ...textInfo, transformer: { context: 64, dModel: 66, heads: 4, blocks: 2 } }
		expect(() => validateTextModelInfo(bad)).toThrow(/Invalid model.json: .*divisible/)
	})

	it("rejects snapshots without a sample", () => {
		const { sample: _s, ...meta } = textInfo.snapshots[0]
		expect(() => validateTextModelInfo({ ...textInfo, snapshots: [meta] })).toThrow(/snapshot 0/)
	})

	it("still rejects transformer files in the image validator", () => {
		expect(() => validateModelInfo(textInfo)).toThrow(/arch/)
	})

	it("keeps validating v1 image files through validateAnyModelInfo", () => {
		expect(validateAnyModelInfo(v1Info).arch).toBe("mlp")
	})

	it("accepts transformer entries in index.json", () => {
		const entry = {
			id: "names",
			arch: "transformer",
			dataset: "names",
			title: "English names",
			path: "names/model.json",
			trainedAt: "2026-10-15T00:00:00.000Z",
			finalAccuracy: 0.3,
		}
		expect(validateIndex([entry])[0].arch).toBe("transformer")
	})
})
