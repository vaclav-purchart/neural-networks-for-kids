import { mkdir, readFile, rm, writeFile } from "node:fs/promises"
import { join } from "node:path"
import type { Dataset, Split } from "../data/load"
import type { Arch, LayerSpec } from "../nn/architectures"
import {
	type ModelIndexEntry,
	type ModelInfo,
	type SnapshotMeta,
	type TextModelInfo,
	type TextSnapshotMeta,
	validateIndex,
	validateModelInfo,
	validateTextModelInfo,
} from "../nn/model-format"
import { type Rand, shuffleInPlace } from "../nn/rng"
import { encodeSnapshot } from "../nn/serialize"
import type { TransformerConfig } from "../nn/transformer/model"

export interface TrainedSnapshot {
	meta: Omit<SnapshotMeta, "file">
	/** Network.params() copies at this point of training. */
	params: Float32Array[]
}

export interface ExportInput {
	outDir: string
	dataset: Pick<Dataset, "spec" | "rows" | "cols" | "mean" | "std">
	arch: Arch
	layers: LayerSpec[]
	snapshots: TrainedSnapshot[]
	samples: { perClass: number; bytes: Uint8Array } | null
}

/** MLP models keep the plain dataset id (and folder) they always had; other architectures get a suffix. */
export function modelId(dataset: string, arch: Arch): string {
	return arch === "mlp" ? dataset : `${dataset}-${arch}`
}

/** Writes outDir/<id>/ (replacing any previous run of the same model) and updates outDir/index.json. */
export async function writeModel(input: ExportInput): Promise<ModelInfo> {
	const { outDir, dataset, arch, layers, snapshots, samples } = input
	const { spec } = dataset
	if (snapshots.length === 0) throw new Error("Nothing to export: no snapshots")
	const id = modelId(spec.id, arch)
	const dir = join(outDir, id)
	await rm(dir, { recursive: true, force: true })
	await mkdir(join(dir, "snapshots"), { recursive: true })

	const metas: SnapshotMeta[] = []
	for (const [i, snap] of snapshots.entries()) {
		const file = `snapshots/${String(i).padStart(3, "0")}.json`
		await writeFile(join(dir, file), JSON.stringify(encodeSnapshot(snap.params)))
		metas.push({ file, ...snap.meta })
	}
	if (samples) await writeFile(join(dir, "samples.bin"), samples.bytes)

	const info = validateModelInfo({
		formatVersion: 2,
		id,
		arch,
		dataset: spec.id,
		title: spec.title,
		labels: spec.labels,
		inputShape: [dataset.rows, dataset.cols],
		layers,
		normalization: { mean: dataset.mean, std: dataset.std },
		preprocessing: { fitBox: spec.fitBox, center: spec.center },
		trainedAt: new Date().toISOString(),
		snapshots: metas,
		samples: samples ? { file: "samples.bin", perClass: samples.perClass } : null,
	} satisfies ModelInfo)
	await writeFile(join(dir, "model.json"), JSON.stringify(info, null, 2))
	await upsertIndex(outDir, {
		id,
		arch,
		dataset: spec.id,
		title: spec.title,
		path: `${id}/model.json`,
		trainedAt: info.trainedAt,
		finalAccuracy: metas[metas.length - 1].testAccuracy,
	})
	return info
}

export interface TextExportInput {
	outDir: string
	id: string
	dataset: string
	title: string
	vocab: string[]
	config: TransformerConfig
	snapshots: { meta: Omit<TextSnapshotMeta, "file">; params: Float32Array[] }[]
}

/** Writes outDir/<id>/ for a text model (replacing any previous run) and updates outDir/index.json. */
export async function writeTextModel(input: TextExportInput): Promise<TextModelInfo> {
	const { outDir, id, dataset, title, vocab, config, snapshots } = input
	if (snapshots.length === 0) throw new Error("Nothing to export: no snapshots")
	const dir = join(outDir, id)
	await rm(dir, { recursive: true, force: true })
	await mkdir(join(dir, "snapshots"), { recursive: true })
	const metas: TextSnapshotMeta[] = []
	for (const [i, snap] of snapshots.entries()) {
		const file = `snapshots/${String(i).padStart(3, "0")}.json`
		await writeFile(join(dir, file), JSON.stringify(encodeSnapshot(snap.params)))
		metas.push({ file, ...snap.meta })
	}
	const info = validateTextModelInfo({
		formatVersion: 2,
		id,
		arch: "transformer",
		dataset,
		title,
		labels: vocab,
		transformer: { context: config.context, dModel: config.dModel, heads: config.heads, blocks: config.blocks },
		trainedAt: new Date().toISOString(),
		snapshots: metas,
	} satisfies TextModelInfo)
	await writeFile(join(dir, "model.json"), JSON.stringify(info, null, 2))
	await upsertIndex(outDir, {
		id,
		arch: "transformer",
		dataset,
		title,
		path: `${id}/model.json`,
		trainedAt: info.trainedAt,
		finalAccuracy: metas[metas.length - 1].testAccuracy,
	})
	return info
}

/** Add or replace this model's entry (by id) in index.json, keeping the others. */
export async function upsertIndex(outDir: string, entry: ModelIndexEntry): Promise<void> {
	const file = join(outDir, "index.json")
	let entries: ModelIndexEntry[] = []
	try {
		entries = validateIndex(JSON.parse(await readFile(file, "utf8")))
	} catch (err) {
		if ((err as NodeJS.ErrnoException).code !== "ENOENT") {
			console.warn(`Replacing unreadable ${file}: ${(err as Error).message}`)
		}
	}
	entries = entries.filter((e) => e.id !== entry.id)
	entries.push(entry)
	entries.sort((a, b) => a.id.localeCompare(b.id))
	await mkdir(outDir, { recursive: true })
	await writeFile(file, JSON.stringify(entries, null, 2))
}

/**
 * Up to perClass random images per class, class-major (all of class 0, then class 1, …).
 * null when some class has no images (e.g. a tiny --limit run).
 */
export function pickSamples(
	split: Split,
	numClasses: number,
	imageSize: number,
	perClass: number,
	rand: Rand,
): { perClass: number; bytes: Uint8Array } | null {
	const byClass: number[][] = Array.from({ length: numClasses }, () => [])
	for (let i = 0; i < split.count; i++) byClass[split.labels[i]]?.push(i)
	const n = Math.min(perClass, ...byClass.map((c) => c.length))
	if (n === 0) return null
	const bytes = new Uint8Array(numClasses * n * imageSize)
	byClass.forEach((indices, c) => {
		const order = Uint32Array.from(indices)
		shuffleInPlace(order, rand)
		for (let k = 0; k < n; k++) {
			const src = order[k] * imageSize
			bytes.set(split.images.subarray(src, src + imageSize), (c * n + k) * imageSize)
		}
	})
	return { perClass: n, bytes }
}
