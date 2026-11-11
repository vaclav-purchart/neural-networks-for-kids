import { buildNetwork } from "../nn/architectures"
import {
	type AnyModelInfo,
	type ModelIndexEntry,
	type ModelInfo,
	type TextModelInfo,
	textConfig,
	validateAnyModelInfo,
	validateIndex,
} from "../nn/model-format"
import { decodeSnapshot } from "../nn/serialize"
import { transformerParamSizes } from "../nn/transformer/model"
import { defaultSource, type ModelSource } from "./model-source"

export interface LoadedModel {
	entry: ModelIndexEntry
	info: AnyModelInfo
	/** Model folder relative to the models root, ending in "/" (or "" at the root). */
	dir: string
	samples: Uint8Array | null
	/** Expected length of every weight tensor, in snapshot order. */
	paramSizes: number[]
}

export type LoadedImageModel = LoadedModel & { info: ModelInfo }
export type LoadedTextModel = LoadedModel & { info: TextModelInfo }

export function isTextModel(model: LoadedModel): model is LoadedTextModel {
	return model.info.arch === "transformer"
}

export function isImageModel(model: LoadedModel): model is LoadedImageModel {
	return model.info.arch !== "transformer"
}

export class ModelStore {
	private readonly snapshots = new Map<string, Promise<Float32Array[]>>()

	constructor(private readonly source: ModelSource = defaultSource()) {}

	private async json(path: string): Promise<unknown> {
		const text = await this.source.text(path)
		if (text === null) throw new Error(`models/${path}: not found`)
		return JSON.parse(text)
	}

	/** Trained models; empty when nothing has been trained yet. */
	async listModels(): Promise<ModelIndexEntry[]> {
		const text = await this.source.text("index.json")
		return text === null ? [] : validateIndex(JSON.parse(text))
	}

	async loadModel(entry: ModelIndexEntry): Promise<LoadedModel> {
		const info = validateAnyModelInfo(await this.json(entry.path))
		const dir = entry.path.slice(0, entry.path.lastIndexOf("/") + 1)
		if (info.arch === "transformer") {
			return { entry, info, dir, samples: null, paramSizes: transformerParamSizes(textConfig(info)) }
		}
		const [rows, cols] = info.inputShape
		const paramSizes = buildNetwork({ c: 1, h: rows, w: cols }, info.layers)
			.params()
			.map((p) => p.length)
		let samples: Uint8Array | null = null
		if (info.samples) {
			const bytes = await this.source.bytes(`${dir}${info.samples.file}`).catch(() => null)
			const expected = info.labels.length * info.samples.perClass * rows * cols
			if (bytes?.length === expected) samples = bytes
			else if (bytes)
				console.warn(`samples.bin has ${bytes.length} bytes, expected ${expected}; sample buttons disabled`)
		}
		return { entry, info, dir, samples, paramSizes }
	}

	/** Decoded weights of one snapshot, cached per training run. */
	loadSnapshot(model: LoadedModel, index: number): Promise<Float32Array[]> {
		const meta = model.info.snapshots[index]
		if (!meta) return Promise.reject(new RangeError(`No snapshot ${index}`))
		const path = `${model.dir}${meta.file}`
		const key = `${path}@${model.info.trainedAt}`
		let pending = this.snapshots.get(key)
		if (!pending) {
			pending = this.json(path).then((json) => decodeSnapshot(json, model.paramSizes))
			pending.catch(() => this.snapshots.delete(key)) // allow a retry after failures
			this.snapshots.set(key, pending)
		}
		return pending
	}
}
