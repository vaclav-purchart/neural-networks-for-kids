import { existsSync } from "node:fs"
import { mkdtemp, readFile, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { describe, expect, it } from "vitest"
import { DATASETS } from "../../src/data/datasets"
import { buildNetwork, cnnSpecs, type LayerSpec, mlpSpecs } from "../../src/nn/architectures"
import { validateModelInfo, validateTextModelInfo } from "../../src/nn/model-format"
import { mulberry32 } from "../../src/nn/rng"
import { decodeSnapshot } from "../../src/nn/serialize"
import { transformerParamSizes } from "../../src/nn/transformer/model"
import { pickSamples, upsertIndex, writeModel, writeTextModel } from "../../src/trainer/export"

const tinySpec = { ...DATASETS.mnist, labels: ["a", "b"], fitBox: 2 }
const tinyDataset = { spec: tinySpec, rows: 2, cols: 2, mean: 0.1, std: 0.3 }
const meta = { imagesSeen: 0, epoch: 0, testAccuracy: 0.5, testSamples: 4, avgLoss: null }

function paramsFor(rows: number, cols: number, layers: LayerSpec[]): Float32Array[] {
	const net = buildNetwork({ c: 1, h: rows, w: cols }, layers)
	net.init(mulberry32(2))
	return net.params()
}

describe("pickSamples", () => {
	const split = { images: Uint8Array.from([0, 0, 1, 1, 0, 0, 1, 1]), labels: Uint8Array.from([0, 1, 0, 1]), count: 4 }

	it("returns class-major samples limited by the smallest class", () => {
		const s = pickSamples(split, 2, 2, 5, mulberry32(1))!
		expect(s.perClass).toBe(2)
		expect([...s.bytes]).toEqual([0, 0, 0, 0, 1, 1, 1, 1]) // image bytes equal their label here
	})

	it("returns null when a class has no images", () => {
		expect(pickSamples(split, 3, 2, 5, mulberry32(1))).toBeNull()
	})
})

describe("writeModel", () => {
	it("writes an MLP under the plain dataset id, with decodable snapshots and an index entry", async () => {
		const outDir = await mkdtemp(join(tmpdir(), "nnvis-"))
		const layers = mlpSpecs([3], 2)
		const params = paramsFor(2, 2, layers)
		const info = await writeModel({
			outDir,
			dataset: tinyDataset,
			arch: "mlp",
			layers,
			snapshots: [{ meta, params }],
			samples: { perClass: 1, bytes: new Uint8Array(8) },
		})
		const dir = join(outDir, "mnist")
		expect(validateModelInfo(JSON.parse(await readFile(join(dir, "model.json"), "utf8")))).toEqual(info)
		expect(info).toMatchObject({
			id: "mnist",
			arch: "mlp",
			preprocessing: { fitBox: 2, center: "mass" },
			samples: { file: "samples.bin", perClass: 1 },
		})
		const snap = JSON.parse(await readFile(join(dir, info.snapshots[0].file), "utf8"))
		expect(
			decodeSnapshot(
				snap,
				params.map((p) => p.length),
			)[0][0],
		).toBeCloseTo(params[0][0], 2)
		expect(existsSync(join(dir, "samples.bin"))).toBe(true)
		const index = JSON.parse(await readFile(join(outDir, "index.json"), "utf8"))
		expect(index).toEqual([
			expect.objectContaining({ id: "mnist", arch: "mlp", path: "mnist/model.json", finalAccuracy: 0.5 }),
		])
	})

	it("writes a CNN to its own folder and id", async () => {
		const outDir = await mkdtemp(join(tmpdir(), "nnvis-"))
		const layers = cnnSpecs([2], [4], 2)
		const info = await writeModel({
			outDir,
			dataset: { ...tinyDataset, rows: 6, cols: 6 },
			arch: "cnn",
			layers,
			snapshots: [{ meta, params: paramsFor(6, 6, layers) }],
			samples: null,
		})
		expect(info).toMatchObject({ id: "mnist-cnn", arch: "cnn", samples: null })
		expect(existsSync(join(outDir, "mnist-cnn", "model.json"))).toBe(true)
	})
})

describe("upsertIndex", () => {
	it("replaces by id and keeps other entries, including ones written before CNN support", async () => {
		const outDir = await mkdtemp(join(tmpdir(), "nnvis-"))
		const legacy = {
			dataset: "mnist",
			title: "MNIST digits",
			path: "mnist/model.json",
			trainedAt: "t",
			finalAccuracy: 0.5,
		}
		await writeFile(join(outDir, "index.json"), JSON.stringify([legacy]))
		const entry = (id: string, arch: "mlp" | "cnn", finalAccuracy: number) => ({
			id,
			arch,
			dataset: "mnist",
			title: "MNIST digits",
			path: `${id}/model.json`,
			trainedAt: "t",
			finalAccuracy,
		})
		await upsertIndex(outDir, entry("mnist-cnn", "cnn", 0.99))
		await upsertIndex(outDir, entry("mnist", "mlp", 0.9))
		const index = JSON.parse(await readFile(join(outDir, "index.json"), "utf8"))
		expect(index.map((e: { id: string; finalAccuracy: number }) => [e.id, e.finalAccuracy])).toEqual([
			["mnist", 0.9],
			["mnist-cnn", 0.99],
		])
	})
})

describe("writeTextModel", () => {
	it("writes snapshots, a valid model.json and an index entry", async () => {
		const outDir = await mkdtemp(join(tmpdir(), "nn-export-text-"))
		const config = { vocab: 3, context: 4, dModel: 8, heads: 2, blocks: 1 }
		const params = transformerParamSizes(config).map((n) => new Float32Array(n).fill(0.25))
		const textMeta = {
			steps: 0,
			charsSeen: 0,
			testLoss: 1.1,
			testAccuracy: 0.3,
			testSamples: 8,
			avgLoss: null,
			sample: "ab\n",
		}
		const info = await writeTextModel({
			outDir,
			id: "names",
			dataset: "names",
			title: "English names",
			vocab: ["\n", "a", "b"],
			config,
			snapshots: [{ meta: textMeta, params }],
		})
		expect(info.arch).toBe("transformer")
		const saved = validateTextModelInfo(JSON.parse(await readFile(join(outDir, "names", "model.json"), "utf8")))
		expect(saved.snapshots[0].file).toBe("snapshots/000.json")
		const snap = JSON.parse(await readFile(join(outDir, "names", "snapshots", "000.json"), "utf8"))
		expect(decodeSnapshot(snap, transformerParamSizes(config))[0][0]).toBe(0.25)
		const index = JSON.parse(await readFile(join(outDir, "index.json"), "utf8"))
		expect(index).toEqual([expect.objectContaining({ id: "names", arch: "transformer", finalAccuracy: 0.3 })])
	})
})
