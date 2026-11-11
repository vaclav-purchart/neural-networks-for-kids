import { existsSync } from "node:fs"
import { readFile } from "node:fs/promises"
import { basename, join } from "node:path"
import { normalizeText } from "../nn/transformer/tokenizer"
import { downloadFile } from "./download"

export type TextDatasetId = "shakespeare" | "names" | "text"

interface RemoteText {
	title: string
	url: string
	fileName: string
}

const REMOTE: Record<Exclude<TextDatasetId, "text">, RemoteText> = {
	shakespeare: {
		title: "Tiny Shakespeare",
		url: "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
		fileName: "input.txt",
	},
	names: {
		title: "English names",
		url: "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt",
		fileName: "names.txt",
	},
}

export const TEXT_DATASET_IDS: TextDatasetId[] = ["shakespeare", "names", "text"]

export function isTextDatasetId(v: string): v is TextDatasetId {
	return (TEXT_DATASET_IDS as string[]).includes(v)
}

export interface TextCorpus {
	/** Model id and folder name. */
	id: string
	title: string
	text: string
}

/** "text-" + the file name without extension, lowercased, accents removed, other runs of symbols → "-". */
export function textModelId(file: string): string {
	const stem = basename(file).replace(/\.[^.]*$/, "")
	const slug = stem
		.normalize("NFD")
		.replace(/\p{M}/gu, "")
		.toLowerCase()
		.replace(/[^a-z0-9]+/g, "-")
		.replace(/^-|-$/g, "")
	return `text-${slug || "file"}`
}

export async function loadTextCorpus(opts: {
	dataset: TextDatasetId
	file?: string
	dataDir: string
	offline: boolean
}): Promise<TextCorpus> {
	if (opts.dataset === "text") {
		const file = opts.file
		if (!file) throw new Error("--dataset text needs --file <path>")
		let raw: string
		try {
			raw = await readFile(file, "utf8")
		} catch (err) {
			throw new Error(`Cannot read ${file}: ${(err as Error).message}`)
		}
		return { id: textModelId(file), title: `Text: ${basename(file)}`, text: normalizeText(raw) }
	}
	const remote = REMOTE[opts.dataset]
	const path = join(opts.dataDir, opts.dataset, remote.fileName)
	if (!existsSync(path)) {
		if (opts.offline) {
			throw new Error(`--offline: missing ${path}\nDownload ${remote.url} and save it as ${path}`)
		}
		console.log(`Downloading ${remote.url}`)
		await downloadFile(remote.url, path)
	}
	return { id: opts.dataset, title: remote.title, text: normalizeText(await readFile(path, "utf8")) }
}
