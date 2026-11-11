import { existsSync, statSync } from "node:fs"
import { mkdtemp } from "node:fs/promises"
import { createServer, type Server } from "node:http"
import type { AddressInfo } from "node:net"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { gzipSync } from "node:zlib"
import { afterEach, describe, expect, it } from "vitest"
import { DATASETS } from "../../src/data/datasets"
import { downloadFile, ensureDatasetFiles } from "../../src/data/download"

let server: Server | null = null
afterEach(() => server?.close())

async function serve(handler: Parameters<typeof createServer>[1]): Promise<string> {
	server = createServer(handler)
	await new Promise<void>((resolve) => server!.listen(0, "127.0.0.1", resolve))
	return `http://127.0.0.1:${(server.address() as AddressInfo).port}/file`
}

describe("downloadFile", () => {
	it("writes the file when the download completes", async () => {
		const url = await serve((_req, res) => {
			res.writeHead(200, { "Content-Length": "1000" })
			res.end(Buffer.alloc(1000, 7))
		})
		const dest = join(await mkdtemp(join(tmpdir(), "nnvis-")), "ok.bin")
		await downloadFile(url, dest)
		expect(statSync(dest).size).toBe(1000)
		expect(existsSync(`${dest}.part`)).toBe(false)
	})

	it("accepts a gzip-encoded response (Content-Length counts compressed bytes)", async () => {
		const body = gzipSync(Buffer.alloc(5000, 65))
		const url = await serve((_req, res) => {
			res.writeHead(200, { "Content-Encoding": "gzip", "Content-Length": String(body.length) })
			res.end(body)
		})
		const dest = join(await mkdtemp(join(tmpdir(), "nnvis-")), "text.txt")
		await downloadFile(url, dest)
		expect(statSync(dest).size).toBe(5000)
	})

	it("leaves nothing behind when the connection drops mid-download", async () => {
		const url = await serve((_req, res) => {
			res.writeHead(200, { "Content-Length": "1000" })
			res.write(Buffer.alloc(100))
			setTimeout(() => res.destroy(), 20)
		})
		const dest = join(await mkdtemp(join(tmpdir(), "nnvis-")), "broken.bin")
		await expect(downloadFile(url, dest)).rejects.toThrow()
		expect(existsSync(dest)).toBe(false)
		expect(existsSync(`${dest}.part`)).toBe(false)
	})

	it("reports HTTP errors with the URL", async () => {
		const url = await serve((_req, res) => {
			res.writeHead(404)
			res.end()
		})
		const dest = join(await mkdtemp(join(tmpdir(), "nnvis-")), "missing.bin")
		await expect(downloadFile(url, dest)).rejects.toThrow(/HTTP 404/)
	})
})

describe("ensureDatasetFiles --offline", () => {
	it("lists the missing files and where to get them", async () => {
		const dataDir = await mkdtemp(join(tmpdir(), "nnvis-"))
		await expect(ensureDatasetFiles(DATASETS.mnist, { dataDir, offline: true, cleanDownloads: false })).rejects.toThrow(
			/trainImages\.idx[\s\S]*ossci-datasets/,
		)
	})
})
