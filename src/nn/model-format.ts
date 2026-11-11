import { type Arch, buildNetwork, type LayerSpec, mlpSpecs } from "./architectures"
import type { Preprocessing } from "./preprocess"
import { type TransformerConfig, validateConfig } from "./transformer/model"

/** Files the trainer writes and the browser reads (see public/models/). */

export interface SnapshotMeta {
	file: string
	imagesSeen: number
	epoch: number
	testAccuracy: number
	/** How many test images testAccuracy was measured on. */
	testSamples: number
	/** Mean training loss since the previous snapshot; null for the initial one. */
	avgLoss: number | null
}

export interface ModelInfo {
	formatVersion: 2
	/** Folder and picker id: the dataset id for MLPs ("mnist"), "<dataset>-cnn" for CNNs. */
	id: string
	arch: Arch
	dataset: string
	title: string
	labels: string[]
	inputShape: [number, number]
	layers: LayerSpec[]
	normalization: { mean: number; std: number }
	/** How the dataset's characters were sized and centred; drawings are prepared the same way. */
	preprocessing: Preprocessing
	trainedAt: string
	snapshots: SnapshotMeta[]
	/** Random test images per class, class-major uint8 pixels. */
	samples: { file: string; perClass: number } | null
}

export interface SnapshotFile {
	params: string[]
}

export interface ModelIndexEntry {
	id: string
	arch: ModelArch
	dataset: string
	title: string
	/** model.json path relative to the models/ folder. */
	path: string
	trainedAt: string
	finalAccuracy: number
}

/** Per-dataset preprocessing for files written before (all of) it was recorded. */
const LEGACY_PREPROCESSING: Record<string, Preprocessing> = {
	mnist: { fitBox: 20, center: "mass" },
	"emnist-letters": { fitBox: 24, center: "box" },
}
const legacyPreprocessing = (dataset: unknown): Preprocessing =>
	LEGACY_PREPROCESSING[String(dataset)] ?? { fitBox: 20, center: "mass" }

export function isObject(v: unknown): v is Record<string, unknown> {
	return typeof v === "object" && v !== null && !Array.isArray(v)
}

const isFiniteNumber = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v)
const isPositiveInt = (v: unknown): v is number => Number.isInteger(v) && (v as number) > 0
const isArch = (v: unknown): v is Arch => v === "mlp" || v === "cnn"
const isModelArch = (v: unknown): v is ModelArch => isArch(v) || v === "transformer"

type Fail = (msg: string) => never

function parseLayerSpec(value: unknown, i: number, fail: Fail): LayerSpec {
	if (!isObject(value)) return fail(`layer ${i} is not an object`)
	switch (value.type) {
		case "dense":
			if (!isPositiveInt(value.units) || (value.activation !== "relu" && value.activation !== "softmax")) {
				return fail(`layer ${i} (dense) needs positive units and activation relu or softmax`)
			}
			return { type: "dense", units: value.units, activation: value.activation }
		case "conv2d":
			if (!isPositiveInt(value.filters) || value.kernel !== 3 || value.activation !== "relu") {
				return fail(`layer ${i} (conv2d) needs positive filters, kernel 3 and activation relu`)
			}
			return { type: "conv2d", filters: value.filters, kernel: 3, activation: "relu" }
		case "maxpool":
			if (value.size !== 2) return fail(`layer ${i} (maxpool) needs size 2`)
			return { type: "maxpool", size: 2 }
		default:
			return fail(`layer ${i} has unknown type ${JSON.stringify(value.type)}`)
	}
}

/** Format v1 (MLP only, `layerSizes`) → v2. */
function upgradeV1(v: Record<string, unknown>, fail: Fail): Record<string, unknown> {
	const sizes = v.layerSizes
	if (!Array.isArray(sizes) || sizes.length < 2 || !sizes.every(isPositiveInt)) {
		return fail("layerSizes must list at least 2 positive integers")
	}
	const shape = v.inputShape
	if (!Array.isArray(shape) || shape.length !== 2 || !shape.every(isPositiveInt) || shape[0] * shape[1] !== sizes[0]) {
		fail(`inputShape must be [rows, cols] with rows × cols = ${sizes[0]}`)
	}
	const { layerSizes: _legacy, ...rest } = v
	return {
		...rest,
		formatVersion: 2,
		id: v.dataset,
		arch: "mlp",
		layers: mlpSpecs(sizes.slice(1, -1), sizes[sizes.length - 1]),
		preprocessing: legacyPreprocessing(v.dataset),
	}
}

export function validateModelInfo(value: unknown): ModelInfo {
	const fail: Fail = (msg) => {
		throw new Error(`Invalid model.json: ${msg}`)
	}
	if (!isObject(value)) return fail("not an object")
	const v = value.formatVersion === 1 ? upgradeV1(value, fail) : value
	if (v.formatVersion !== 2) fail(`unsupported formatVersion ${String(v.formatVersion)}`)
	if (typeof v.id !== "string" || typeof v.dataset !== "string" || typeof v.title !== "string") {
		fail("id, dataset and title must be strings")
	}
	if (!isArch(v.arch)) fail('arch must be "mlp" or "cnn"')
	if (!Array.isArray(v.labels) || v.labels.length < 2 || !v.labels.every((l) => typeof l === "string")) {
		fail("labels must be a list of at least 2 strings")
	}
	const labels = v.labels as string[]
	const shape = v.inputShape
	if (!Array.isArray(shape) || shape.length !== 2 || !shape.every(isPositiveInt))
		fail("inputShape must be [rows, cols]")
	const [rows, cols] = shape as [number, number]
	if (!Array.isArray(v.layers) || v.layers.length === 0) fail("layers must be a non-empty list")
	const layers = (v.layers as unknown[]).map((l, i) => parseLayerSpec(l, i, fail))
	const last = layers[layers.length - 1]
	if (last.type !== "dense" || last.activation !== "softmax") fail("the last layer must be dense with softmax")
	const outputUnits = (last as { units: number }).units
	if (outputUnits !== labels.length) fail(`${labels.length} labels but output layer has ${outputUnits} neurons`)
	layers.slice(0, -1).forEach((l, i) => {
		if (l.type === "dense" && l.activation !== "relu") fail(`layer ${i} (dense): hidden layers must use relu`)
	})
	try {
		buildNetwork({ c: 1, h: rows, w: cols }, layers)
	} catch (err) {
		fail((err as Error).message)
	}
	const norm = v.normalization
	if (!isObject(norm) || !isFiniteNumber(norm.mean) || !isFiniteNumber(norm.std) || norm.std <= 0) {
		fail("normalization must be { mean, std } with std > 0")
	}
	const pre = v.preprocessing
	if (!isObject(pre) || !isPositiveInt(pre.fitBox) || pre.fitBox > Math.min(rows, cols)) {
		fail("preprocessing.fitBox must be a positive integer no larger than the image")
	}
	const center = (pre as Record<string, unknown>).center ?? legacyPreprocessing(v.dataset).center
	if (center !== "mass" && center !== "box") fail('preprocessing.center must be "mass" or "box"')
	const preprocessing: Preprocessing = {
		fitBox: (pre as { fitBox: number }).fitBox,
		center: center as Preprocessing["center"],
	}
	if (!Array.isArray(v.snapshots) || v.snapshots.length === 0) fail("snapshots must be a non-empty list")
	;(v.snapshots as unknown[]).forEach((s, i) => {
		const ok =
			isObject(s) &&
			typeof s.file === "string" &&
			isFiniteNumber(s.imagesSeen) &&
			isFiniteNumber(s.epoch) &&
			isFiniteNumber(s.testAccuracy) &&
			isFiniteNumber(s.testSamples) &&
			(s.avgLoss === null || isFiniteNumber(s.avgLoss))
		if (!ok) fail(`snapshot ${i} is malformed`)
	})
	const samples = v.samples
	if (samples !== null && !(isObject(samples) && typeof samples.file === "string" && isPositiveInt(samples.perClass))) {
		fail("samples must be null or { file, perClass }")
	}
	return { ...v, layers, preprocessing } as unknown as ModelInfo
}

export function validateIndex(value: unknown): ModelIndexEntry[] {
	if (!Array.isArray(value)) throw new Error("Invalid models/index.json: expected an array")
	return value.map((e, i) => {
		const ok =
			isObject(e) &&
			typeof e.dataset === "string" &&
			typeof e.title === "string" &&
			typeof e.path === "string" &&
			typeof e.trainedAt === "string" &&
			isFiniteNumber(e.finalAccuracy) &&
			(e.id === undefined || typeof e.id === "string") &&
			(e.arch === undefined || isModelArch(e.arch))
		if (!ok) throw new Error(`Invalid models/index.json: entry ${i} is malformed`)
		// Entries written before CNN support have no id / arch: they are MLPs named after their dataset.
		return { ...e, id: e.id ?? e.dataset, arch: e.arch ?? "mlp" } as ModelIndexEntry
	})
}

/** Image architectures plus the character-level transformer. */
export type ModelArch = Arch | "transformer"

export interface TextSnapshotMeta {
	file: string
	/** Optimizer steps so far. */
	steps: number
	/** Training characters seen (steps × batch × context). */
	charsSeen: number
	testLoss: number
	/** Fraction of test positions where the most likely next character was right. */
	testAccuracy: number
	/** How many test positions testLoss / testAccuracy were measured on. */
	testSamples: number
	/** Mean training loss since the previous snapshot; null for the initial one. */
	avgLoss: number | null
	/** Text generated at train time from a fixed prompt and seed. */
	sample: string
}

export interface TextModelInfo {
	formatVersion: 2
	id: string
	arch: "transformer"
	dataset: string
	title: string
	/** The vocabulary: one entry per token id. */
	labels: string[]
	transformer: { context: number; dModel: number; heads: number; blocks: number }
	trainedAt: string
	snapshots: TextSnapshotMeta[]
}

export type AnyModelInfo = ModelInfo | TextModelInfo

export function textConfig(info: TextModelInfo): TransformerConfig {
	return { vocab: info.labels.length, ...info.transformer }
}

export function validateTextModelInfo(value: unknown): TextModelInfo {
	const fail: Fail = (msg) => {
		throw new Error(`Invalid model.json: ${msg}`)
	}
	if (!isObject(value)) return fail("not an object")
	const v = value
	if (v.formatVersion !== 2) fail(`unsupported formatVersion ${String(v.formatVersion)}`)
	if (v.arch !== "transformer") fail('arch must be "transformer"')
	if (typeof v.id !== "string" || typeof v.dataset !== "string" || typeof v.title !== "string") {
		fail("id, dataset and title must be strings")
	}
	if (!Array.isArray(v.labels) || v.labels.length < 2 || !v.labels.every((l) => typeof l === "string")) {
		fail("labels must be a list of at least 2 strings")
	}
	const t = v.transformer
	if (!isObject(t)) return fail("transformer must be { context, dModel, heads, blocks }")
	const config = {
		vocab: (v.labels as string[]).length,
		context: t.context as number,
		dModel: t.dModel as number,
		heads: t.heads as number,
		blocks: t.blocks as number,
	}
	try {
		validateConfig(config)
	} catch (err) {
		fail((err as Error).message)
	}
	if (typeof v.trainedAt !== "string") fail("trainedAt must be a string")
	if (!Array.isArray(v.snapshots) || v.snapshots.length === 0) fail("snapshots must be a non-empty list")
	;(v.snapshots as unknown[]).forEach((s, i) => {
		const ok =
			isObject(s) &&
			typeof s.file === "string" &&
			isFiniteNumber(s.steps) &&
			isFiniteNumber(s.charsSeen) &&
			isFiniteNumber(s.testLoss) &&
			isFiniteNumber(s.testAccuracy) &&
			isFiniteNumber(s.testSamples) &&
			(s.avgLoss === null || isFiniteNumber(s.avgLoss)) &&
			typeof s.sample === "string"
		if (!ok) fail(`snapshot ${i} is malformed`)
	})
	const { vocab: _vocab, ...transformer } = config
	return { ...(v as unknown as TextModelInfo), transformer }
}

/** Image (v1/v2) or text model.json. */
export function validateAnyModelInfo(value: unknown): AnyModelInfo {
	return isObject(value) && value.arch === "transformer" ? validateTextModelInfo(value) : validateModelInfo(value)
}
