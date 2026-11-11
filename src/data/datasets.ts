import type { Centering } from "../nn/preprocess"

export type DatasetId = "mnist" | "emnist-letters"
export type SplitFile = "trainImages" | "trainLabels" | "testImages" | "testLabels"
export const SPLIT_FILES: readonly SplitFile[] = ["trainImages", "trainLabels", "testImages", "testLabels"]

export type DatasetSource =
	| { kind: "gzip-files"; urls: Record<SplitFile, string> }
	| { kind: "zip-archive"; url: string; archiveName: string; members: Record<SplitFile, string> }

export interface DatasetSpec {
	id: DatasetId
	title: string
	labels: string[]
	source: DatasetSource
	/** Added to every raw label (EMNIST letters are 1..26). */
	labelOffset: number
	/** EMNIST stores images transposed relative to MNIST. */
	transpose: boolean
	/** Character size in the images: drawings are scaled so their ink fits fitBox × fitBox. */
	fitBox: number
	/** How the dataset centred its characters (MNIST: centre of mass; EMNIST: bounding box). */
	center: Centering
}

const MNIST_BASE = "https://ossci-datasets.s3.amazonaws.com/mnist/"

export const DATASETS: Record<DatasetId, DatasetSpec> = {
	mnist: {
		id: "mnist",
		title: "MNIST digits",
		labels: [..."0123456789"],
		source: {
			kind: "gzip-files",
			urls: {
				trainImages: `${MNIST_BASE}train-images-idx3-ubyte.gz`,
				trainLabels: `${MNIST_BASE}train-labels-idx1-ubyte.gz`,
				testImages: `${MNIST_BASE}t10k-images-idx3-ubyte.gz`,
				testLabels: `${MNIST_BASE}t10k-labels-idx1-ubyte.gz`,
			},
		},
		labelOffset: 0,
		transpose: false,
		fitBox: 20,
		center: "mass",
	},
	"emnist-letters": {
		id: "emnist-letters",
		title: "EMNIST letters",
		labels: [..."ABCDEFGHIJKLMNOPQRSTUVWXYZ"],
		source: {
			kind: "zip-archive",
			url: "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip",
			archiveName: "emnist-gzip.zip",
			members: {
				trainImages: "gzip/emnist-letters-train-images-idx3-ubyte.gz",
				trainLabels: "gzip/emnist-letters-train-labels-idx1-ubyte.gz",
				testImages: "gzip/emnist-letters-test-images-idx3-ubyte.gz",
				testLabels: "gzip/emnist-letters-test-labels-idx1-ubyte.gz",
			},
		},
		labelOffset: -1,
		transpose: true,
		fitBox: 24,
		center: "box",
	},
}

export const DATASET_IDS = Object.keys(DATASETS) as DatasetId[]

export function isDatasetId(v: string): v is DatasetId {
	return Object.hasOwn(DATASETS, v)
}
