import { mkdtemp, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { describe, expect, it } from "vitest"
import { isTextDatasetId, loadTextCorpus, textModelId } from "../../src/data/text"

describe("text datasets", () => {
	it("knows the text dataset ids", () => {
		expect(isTextDatasetId("shakespeare")).toBe(true)
		expect(isTextDatasetId("names")).toBe(true)
		expect(isTextDatasetId("text")).toBe(true)
		expect(isTextDatasetId("mnist")).toBe(false)
	})

	it("derives the model id from the file name", () => {
		expect(textModelId("/books/Babička (1855).txt")).toBe("text-babicka-1855")
	})

	it("reads and normalizes a local --file", async () => {
		const dir = await mkdtemp(join(tmpdir(), "nn-text-"))
		const file = join(dir, "book.txt")
		await writeFile(file, "﻿ahoj\r\nsvěte\r\n")
		const corpus = await loadTextCorpus({ dataset: "text", file, dataDir: dir, offline: true })
		expect(corpus).toEqual({ id: "text-book", title: "Text: book.txt", text: "ahoj\nsvěte\n" })
	})

	it("explains a missing --file", async () => {
		await expect(
			loadTextCorpus({ dataset: "text", file: "/nope/missing.txt", dataDir: "data", offline: true }),
		).rejects.toThrow(/Cannot read \/nope\/missing.txt/)
	})

	it("explains what to download when offline", async () => {
		const dir = await mkdtemp(join(tmpdir(), "nn-text-"))
		await expect(loadTextCorpus({ dataset: "names", dataDir: dir, offline: true })).rejects.toThrow(
			/--offline.*names\.txt/s,
		)
	})
})
