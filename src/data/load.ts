import { readFile } from "node:fs/promises"
import { DATASETS, type DatasetId, type DatasetSpec, SPLIT_FILES } from "./datasets"
import { type EnsureOptions, ensureDatasetFiles } from "./download"
import { type IdxArray, parseIdx } from "./idx"
import { remapLabels, transposeImages } from "./transform"

/** Images stay uint8 (0..255) to save memory; the trainer normalises per batch. */
export interface Split {
	images: Uint8Array
	labels: Uint8Array
	count: number
}

export interface Dataset {
	spec: DatasetSpec
	rows: number
	cols: number
	train: Split
	test: Split
	/** Mean / std of training pixels scaled to 0..1. */
	mean: number
	std: number
}

export interface LoadOptions extends EnsureOptions {
	/** Use only the first N images of each split. */
	limit?: number
}

export async function loadDataset(id: DatasetId, opts: LoadOptions): Promise<Dataset> {
	const spec = DATASETS[id]
	const paths = await ensureDatasetFiles(spec, opts)
	const [trainImages, trainLabels, testImages, testLabels] = await Promise.all(
		SPLIT_FILES.map(async (k) => parseIdx(await readFile(paths[k]), paths[k])),
	)
	if (trainImages.shape.length !== 3 || testImages.shape.length !== 3) throw new Error("Image files must be 3-D")
	const [, rows, cols] = trainImages.shape
	if (testImages.shape[1] !== rows || testImages.shape[2] !== cols) throw new Error("Train and test image sizes differ")
	const train = makeSplit(spec, trainImages, trainLabels, rows, cols, opts.limit)
	const test = makeSplit(spec, testImages, testLabels, rows, cols, opts.limit)
	return { spec, rows, cols, train, test, ...pixelStats(train.images) }
}

export function makeSplit(
	spec: DatasetSpec,
	images: IdxArray,
	labels: IdxArray,
	rows: number,
	cols: number,
	limit?: number,
): Split {
	if (images.shape[0] !== labels.shape[0]) {
		throw new Error(`${images.shape[0]} images but ${labels.shape[0]} labels`)
	}
	const count = Math.min(images.shape[0], limit ?? Infinity)
	const imgs = images.data.slice(0, count * rows * cols) // copy, so transforms never touch the source
	if (spec.transpose) transposeImages(imgs, rows, cols)
	return {
		images: imgs,
		labels: remapLabels(labels.data.subarray(0, count), spec.labelOffset, spec.labels.length),
		count,
	}
}

export function pixelStats(images: Uint8Array): { mean: number; std: number } {
	let sum = 0
	let sumSq = 0
	for (let i = 0; i < images.length; i++) {
		const p = images[i] / 255
		sum += p
		sumSq += p * p
	}
	const mean = sum / images.length
	return { mean, std: Math.sqrt(Math.max(sumSq / images.length - mean * mean, 1e-12)) }
}
