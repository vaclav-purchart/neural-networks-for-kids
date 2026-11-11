import { describe, expect, it } from "vitest"
import { buildNetwork, cnnSpecs, mlpSpecs } from "../../src/nn/architectures"
import { encodeFloat16 } from "../../src/nn/float16"
import { validateIndex, validateModelInfo } from "../../src/nn/model-format"
import { mulberry32 } from "../../src/nn/rng"
import { decodeSnapshot, encodeSnapshot } from "../../src/nn/serialize"

const snapshots = [
	{ file: "snapshots/000.json", imagesSeen: 0, epoch: 0, testAccuracy: 0.5, testSamples: 10, avgLoss: null },
]

const v1Info = () => ({
	formatVersion: 1,
	dataset: "mnist",
	title: "MNIST digits",
	labels: ["a", "b"],
	inputShape: [28, 28],
	layerSizes: [784, 3, 2],
	normalization: { mean: 0.1, std: 0.3 },
	trainedAt: "2026-10-12T00:00:00.000Z",
	snapshots,
	samples: null,
})

/** A v2 file as written before preprocessing.center existed. */
const v2Info = () => ({
	formatVersion: 2,
	id: "mnist-cnn",
	arch: "cnn",
	dataset: "mnist",
	title: "MNIST digits",
	labels: [..."0123456789"],
	inputShape: [28, 28],
	layers: cnnSpecs([8, 16], [64], 10),
	normalization: { mean: 0.1, std: 0.3 },
	preprocessing: { fitBox: 20 },
	trainedAt: "2026-10-13T00:00:00.000Z",
	snapshots,
	samples: null,
})

describe("snapshot serialisation", () => {
	it("round-trips every parameter tensor within float16 tolerance", () => {
		const net = buildNetwork({ c: 1, h: 6, w: 6 }, cnnSpecs([2], [4], 3))
		net.init(mulberry32(1))
		const params = net.params()
		const json = JSON.parse(JSON.stringify(encodeSnapshot(params)))
		const decoded = decodeSnapshot(
			json,
			params.map((p) => p.length),
		)
		decoded.forEach((t, i) => {
			t.forEach((v, j) => {
				expect(v).toBeCloseTo(params[i][j], 2)
			})
		})
	})

	it("still reads v1 snapshots ({ layers: [{ W, b }] })", () => {
		const v1 = { layers: [{ W: encodeFloat16([1, 2]), b: encodeFloat16([3]) }] }
		expect(decodeSnapshot(v1, [2, 1]).map((t) => [...t])).toEqual([[1, 2], [3]])
	})

	it("rejects snapshots that don't match the network", () => {
		const snap = encodeSnapshot([new Float32Array(12), new Float32Array(3)])
		expect(() => decodeSnapshot(snap, [15, 3])).toThrow(/tensor 0 has 12 values, expected 15/)
		expect(() => decodeSnapshot(snap, [12])).toThrow(/expected 1 tensors, got 2/)
		expect(() => decodeSnapshot({ nope: true }, [1])).toThrow(/missing params/)
	})
})

describe("validateModelInfo", () => {
	it("accepts a v2 CNN model", () => {
		expect(validateModelInfo(v2Info()).arch).toBe("cnn")
	})

	it("fills in the dataset's centring for v2 files written before it was recorded", () => {
		expect(validateModelInfo(v2Info()).preprocessing).toEqual({ fitBox: 20, center: "mass" })
		const letters = { ...v2Info(), dataset: "emnist-letters", labels: [..."ABCDEFGHIJ"], preprocessing: { fitBox: 24 } }
		expect(validateModelInfo(letters).preprocessing).toEqual({ fitBox: 24, center: "box" })
		expect(validateModelInfo({ ...v2Info(), preprocessing: { fitBox: 20, center: "box" } }).preprocessing.center).toBe(
			"box",
		)
		expect(() => validateModelInfo({ ...v2Info(), preprocessing: { fitBox: 20, center: "middle" } })).toThrow(
			/preprocessing.center/,
		)
	})

	it("upgrades v1 MLP models", () => {
		const info = validateModelInfo(v1Info())
		expect(info).toMatchObject({
			formatVersion: 2,
			id: "mnist",
			arch: "mlp",
			preprocessing: { fitBox: 20, center: "mass" },
		})
		expect(info.layers).toEqual(mlpSpecs([3], 2))
		expect("layerSizes" in info).toBe(false)
		expect(validateModelInfo({ ...v1Info(), dataset: "emnist-letters" }).preprocessing).toEqual({
			fitBox: 24,
			center: "box",
		})
	})

	it("names the problem for invalid models", () => {
		expect(() => validateModelInfo({ ...v1Info(), labels: ["a", "b", "c"] })).toThrow(/3 labels but output layer has 2/)
		expect(() => validateModelInfo({ ...v1Info(), inputShape: [3, 3] })).toThrow(/inputShape/)
		expect(() => validateModelInfo({ ...v1Info(), snapshots: [] })).toThrow(/snapshots/)
		expect(() => validateModelInfo({ ...v1Info(), normalization: { mean: 0, std: 0 } })).toThrow(/normalization/)
		expect(() => validateModelInfo("garbage")).toThrow(/not an object/)
		expect(() => validateModelInfo({ ...v2Info(), layers: [{ type: "lstm" }] })).toThrow(
			/layer 0 has unknown type "lstm"/,
		)
		expect(() => validateModelInfo({ ...v2Info(), layers: cnnSpecs([8, 8, 8, 8], [64], 10) })).toThrow(
			/layer 6 \(conv2d\)/,
		)
		expect(() =>
			validateModelInfo({
				...v2Info(),
				layers: [...mlpSpecs([64], 10).slice(0, 1), { type: "maxpool", size: 2 }, ...mlpSpecs([], 10)],
			}),
		).toThrow(/layer 1 \(maxpool\)/)
		expect(() => validateModelInfo({ ...v2Info(), preprocessing: { fitBox: 40 } })).toThrow(/fitBox/)
		expect(() => validateModelInfo({ ...v2Info(), arch: "rnn" })).toThrow(/arch/)
	})
})

describe("validateIndex", () => {
	it("gives entries written before CNN support an id and arch", () => {
		const legacy = { dataset: "mnist", title: "MNIST", path: "mnist/model.json", trainedAt: "x", finalAccuracy: 0.9 }
		expect(validateIndex([legacy])).toEqual([{ ...legacy, id: "mnist", arch: "mlp" }])
		expect(() => validateIndex({})).toThrow(/array/)
		expect(() => validateIndex([{ dataset: 1 }])).toThrow(/entry 0/)
	})
})
