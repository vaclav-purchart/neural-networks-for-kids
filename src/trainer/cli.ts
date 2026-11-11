import { parseArgs } from "node:util"
import { DATASET_IDS, type DatasetId, isDatasetId } from "../data/datasets"
import { isTextDatasetId, TEXT_DATASET_IDS, type TextDatasetId } from "../data/text"
import type { Arch } from "../nn/architectures"
import { validateConfig } from "../nn/transformer/model"

export interface ImageTrainOptions {
	kind: "image"
	dataset: DatasetId
	arch: Arch
	hidden: number[]
	/** Conv filters per conv+pool block (CNN only; empty for MLP). */
	filters: number[]
	epochs: number
	batch: number
	lr: number
	seed: number
	limit?: number
	samplesPerClass: number
	offline: boolean
	cleanDownloads: boolean
	dataDir: string
	outDir: string
}

export interface TextTrainOptions {
	kind: "text"
	dataset: TextDatasetId
	/** Local text file (dataset "text" only). */
	file?: string
	context: number
	dModel: number
	heads: number
	blocks: number
	steps: number
	batch: number
	lr: number
	seed: number
	offline: boolean
	dataDir: string
	outDir: string
}

export type TrainOptions = ImageTrainOptions | TextTrainOptions

/** Chosen so shakespeare trains in roughly 10–20 minutes on an M-series Mac. */
export const DEFAULT_TEXT_STEPS = 1000

const ALL_DATASETS = [...DATASET_IDS, ...TEXT_DATASET_IDS]

export const USAGE = `Usage: npm run train -- --dataset <${ALL_DATASETS.join("|")}> [options]

Image datasets (${DATASET_IDS.join(", ")}):
  --arch mlp               mlp (dense layers) or cnn (convolutional)
  --hidden 128,64          dense hidden layer sizes (cnn default: 64)
  --filters 8,16           cnn only: filters per conv+pool block
  --epochs 5               passes over the training set (cnn default: 3)
  --batch 64               mini-batch size
  --lr 0.001               Adam learning rate
  --limit N                use only the first N train/test images (quick experiments)
  --samples-per-class 100  test images shipped to the browser per class
  --clean-downloads        delete the large EMNIST zip after extracting

Text datasets (${TEXT_DATASET_IDS.join(", ")}) train a character-level transformer:
  --file book.txt          the text to learn (dataset "text" only)
  --context 64             characters the model sees at once
  --d-model 64             numbers per character inside the model
  --heads 4                attention heads per block (must divide --d-model)
  --blocks 2               transformer blocks
  --steps ${DEFAULT_TEXT_STEPS}             training steps
  --batch 32               text windows per step
  --lr 0.003               Adam learning rate (decays to 10% over the run)

Both:
  --seed 42                random seed (init + shuffling)
  --offline                never download; explain what is missing instead`

const IMAGE_ONLY = ["hidden", "filters", "epochs", "limit", "samples-per-class", "clean-downloads"] as const
const TEXT_ONLY = ["file", "context", "d-model", "heads", "blocks", "steps"] as const

function positiveInt(text: string, name: string): number {
	const n = Number(text)
	if (!Number.isInteger(n) || n <= 0) throw new Error(`${name} must be a positive integer, got "${text}"`)
	return n
}

function parseSizes(text: string, name: string): number[] {
	const sizes = text
		.split(",")
		.map((s) => s.trim())
		.filter(Boolean)
		.map((s) => positiveInt(s, name))
	if (sizes.length === 0) throw new Error(`${name} needs at least one size`)
	return sizes
}

function positiveNumber(text: string, name: string): number {
	const n = Number(text)
	if (!Number.isFinite(n) || n <= 0) throw new Error(`${name} must be a positive number, got "${text}"`)
	return n
}

export function parseTrainArgs(argv: string[]): TrainOptions {
	const { values } = parseArgs({
		args: argv,
		strict: true,
		allowPositionals: false,
		options: {
			dataset: { type: "string" },
			arch: { type: "string" },
			filters: { type: "string" },
			hidden: { type: "string" },
			epochs: { type: "string" },
			batch: { type: "string" },
			lr: { type: "string" },
			seed: { type: "string" },
			limit: { type: "string" },
			"samples-per-class": { type: "string" },
			offline: { type: "boolean" },
			"clean-downloads": { type: "boolean" },
			"data-dir": { type: "string" },
			"out-dir": { type: "string" },
			file: { type: "string" },
			context: { type: "string" },
			"d-model": { type: "string" },
			heads: { type: "string" },
			blocks: { type: "string" },
			steps: { type: "string" },
		},
	})
	const dataset = values.dataset ?? ""
	const seed = Number(values.seed ?? "42")
	if (!Number.isInteger(seed) || seed < 0)
		throw new Error(`--seed must be a non-negative integer, got "${values.seed}"`)
	const common = {
		seed,
		offline: values.offline ?? false,
		dataDir: values["data-dir"] ?? "data",
		outDir: values["out-dir"] ?? "public/models",
	}

	if (isTextDatasetId(dataset)) {
		for (const flag of IMAGE_ONLY) {
			if (values[flag] !== undefined) throw new Error(`--${flag} only applies to image datasets`)
		}
		if (values.arch !== undefined && values.arch !== "transformer") {
			throw new Error(`Text datasets train a transformer; --arch ${values.arch} is not available`)
		}
		if (dataset === "text" && !values.file) throw new Error("--dataset text needs --file <path>")
		if (dataset !== "text" && values.file !== undefined) throw new Error("--file only applies to --dataset text")
		const options: TextTrainOptions = {
			kind: "text",
			dataset,
			file: values.file,
			context: positiveInt(values.context ?? "64", "--context"),
			dModel: positiveInt(values["d-model"] ?? "64", "--d-model"),
			heads: positiveInt(values.heads ?? "4", "--heads"),
			blocks: positiveInt(values.blocks ?? "2", "--blocks"),
			steps: positiveInt(values.steps ?? String(DEFAULT_TEXT_STEPS), "--steps"),
			batch: positiveInt(values.batch ?? "32", "--batch"),
			lr: positiveNumber(values.lr ?? "0.003", "--lr"),
			...common,
		}
		const { context, dModel, heads, blocks } = options
		validateConfig({ vocab: 2, context, dModel, heads, blocks })
		return options
	}

	if (!isDatasetId(dataset)) throw new Error(`--dataset must be one of: ${ALL_DATASETS.join(", ")}`)
	for (const flag of TEXT_ONLY) {
		if (values[flag] !== undefined) throw new Error(`--${flag} only applies to text datasets`)
	}
	const arch = values.arch ?? "mlp"
	if (arch === "transformer") throw new Error("--arch transformer needs a text dataset (e.g. --dataset shakespeare)")
	if (arch !== "mlp" && arch !== "cnn") throw new Error(`--arch must be mlp or cnn, got "${arch}"`)
	if (arch === "mlp" && values.filters !== undefined) throw new Error("--filters only applies to --arch cnn")
	return {
		kind: "image",
		dataset,
		arch,
		hidden: parseSizes(values.hidden ?? (arch === "cnn" ? "64" : "128,64"), "--hidden"),
		filters: arch === "cnn" ? parseSizes(values.filters ?? "8,16", "--filters") : [],
		epochs: positiveInt(values.epochs ?? (arch === "cnn" ? "3" : "5"), "--epochs"),
		batch: positiveInt(values.batch ?? "64", "--batch"),
		lr: positiveNumber(values.lr ?? "0.001", "--lr"),
		limit: values.limit === undefined ? undefined : positiveInt(values.limit, "--limit"),
		samplesPerClass: positiveInt(values["samples-per-class"] ?? "100", "--samples-per-class"),
		cleanDownloads: values["clean-downloads"] ?? false,
		...common,
	}
}
