import { describe, expect, it } from "vitest"
import { encodeSnapshot } from "../../src/nn/serialize"
import { transformerParamSizes } from "../../src/nn/transformer/model"
import { embeddedSource } from "../../src/web/model-source"
import { isTextModel, ModelStore } from "../../src/web/model-store"

const binary = (bytes: number[]) => ({ bin: Buffer.from(bytes).toString("base64") })

describe("embeddedSource", () => {
	it("returns text files as stored and null for missing paths", async () => {
		const source = embeddedSource({ "index.json": "[]" })
		expect(await source.text("index.json")).toBe("[]")
		expect(await source.text("nope.json")).toBeNull()
	})

	it("decodes binary files", async () => {
		const source = embeddedSource({ "m/samples.bin": binary([0, 7, 255]) })
		expect(await source.bytes("m/samples.bin")).toEqual(new Uint8Array([0, 7, 255]))
		expect(await source.bytes("m/other.bin")).toBeNull()
	})
})

describe("ModelStore with an embedded source", () => {
	it("lists no models when index.json is absent", async () => {
		expect(await new ModelStore(embeddedSource({})).listModels()).toEqual([])
	})

	it("loads a model, its samples and a snapshot without fetching", async () => {
		const info = {
			formatVersion: 2,
			id: "tiny",
			arch: "mlp",
			dataset: "mnist",
			title: "Tiny",
			labels: ["a", "b"],
			inputShape: [28, 28],
			layers: [{ type: "dense", units: 2, activation: "softmax" }],
			normalization: { mean: 0, std: 1 },
			preprocessing: { fitBox: 20, center: "mass" },
			trainedAt: "2026-10-13T00:00:00.000Z",
			snapshots: [
				{ file: "snapshots/000.json", imagesSeen: 0, epoch: 0, testAccuracy: 0.5, testSamples: 10, avgLoss: null },
			],
			samples: { file: "samples.bin", perClass: 1 },
		}
		const index = [
			{
				id: "tiny",
				arch: "mlp",
				dataset: "mnist",
				title: "Tiny",
				path: "tiny/model.json",
				trainedAt: info.trainedAt,
				finalAccuracy: 0.5,
			},
		]
		const store = new ModelStore(
			embeddedSource({
				"index.json": JSON.stringify(index),
				"tiny/model.json": JSON.stringify(info),
				"tiny/samples.bin": binary(new Array(2 * 28 * 28).fill(9)),
				"tiny/snapshots/000.json": JSON.stringify(
					encodeSnapshot([new Float32Array(28 * 28 * 2).fill(0.5), new Float32Array([1, -1])]),
				),
			}),
		)
		const [entry] = await store.listModels()
		const loaded = await store.loadModel(entry)
		expect(loaded.samples?.length).toBe(2 * 28 * 28)
		expect(loaded.paramSizes).toEqual([28 * 28 * 2, 2])
		const [W, b] = await store.loadSnapshot(loaded, 0)
		expect(W[0]).toBe(0.5)
		expect(Array.from(b)).toEqual([1, -1])
	})
	it("loads a text model with transformer parameter sizes and no samples", async () => {
		const config = { context: 4, dModel: 8, heads: 2, blocks: 1 }
		const info = {
			formatVersion: 2,
			id: "names",
			arch: "transformer",
			dataset: "names",
			title: "English names",
			labels: ["\n", "a", "b"],
			transformer: config,
			trainedAt: "2026-10-15T00:00:00.000Z",
			snapshots: [
				{
					file: "snapshots/000.json",
					steps: 0,
					charsSeen: 0,
					testLoss: 1,
					testAccuracy: 0.3,
					testSamples: 8,
					avgLoss: null,
					sample: "ab",
				},
			],
		}
		const entry = {
			id: "names",
			arch: "transformer",
			dataset: "names",
			title: "English names",
			path: "names/model.json",
			trainedAt: info.trainedAt,
			finalAccuracy: 0.3,
		}
		const store = new ModelStore(
			embeddedSource({ "index.json": JSON.stringify([entry]), "names/model.json": JSON.stringify(info) }),
		)
		const [listed] = await store.listModels()
		const loaded = await store.loadModel(listed)
		expect(isTextModel(loaded)).toBe(true)
		expect(loaded.samples).toBeNull()
		expect(loaded.paramSizes).toEqual(transformerParamSizes({ vocab: 3, ...config }))
	})
})
