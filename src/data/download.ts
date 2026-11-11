import { createReadStream, createWriteStream, existsSync } from "node:fs"
import { mkdir, rename, rm } from "node:fs/promises"
import { dirname, join } from "node:path"
import { Readable, Transform } from "node:stream"
import { pipeline } from "node:stream/promises"
import type { ReadableStream as NodeReadableStream } from "node:stream/web"
import { createGunzip } from "node:zlib"
import yauzl from "yauzl"
import { type DatasetSpec, SPLIT_FILES, type SplitFile } from "./datasets"

export interface EnsureOptions {
	dataDir: string
	offline: boolean
	cleanDownloads: boolean
}

const mb = (bytes: number) => (bytes / 1e6).toFixed(1)

/** Download url → dest. Writes to dest.part first, so dest only exists once complete. */
export async function downloadFile(url: string, dest: string): Promise<void> {
	await mkdir(dirname(dest), { recursive: true })
	const part = `${dest}.part`
	const res = await fetch(url)
	if (!res.ok || !res.body) throw new Error(`Download failed: ${url} → HTTP ${res.status}`)
	// With Content-Encoding (e.g. gzip) fetch decompresses, so Content-Length isn't the size we receive.
	const encoded = (res.headers.get("content-encoding") ?? "identity") !== "identity"
	const total = encoded ? 0 : Number(res.headers.get("content-length")) || 0
	let received = 0
	let lastLog = Date.now()
	const progress = new Transform({
		transform(chunk: Buffer, _enc, callback) {
			received += chunk.length
			if (Date.now() - lastLog > 2000) {
				lastLog = Date.now()
				console.log(`  ${mb(received)}${total ? ` / ${mb(total)}` : ""} MB`)
			}
			callback(null, chunk)
		},
	})
	try {
		await pipeline(Readable.fromWeb(res.body as NodeReadableStream), progress, createWriteStream(part))
		if (total && received !== total) throw new Error(`Download incomplete: ${url} (${received} of ${total} bytes)`)
	} catch (err) {
		await rm(part, { force: true })
		throw err
	}
	await rename(part, dest)
}

async function gunzipFile(src: string, dest: string): Promise<void> {
	const part = `${dest}.part`
	try {
		await pipeline(createReadStream(src), createGunzip(), createWriteStream(part))
	} catch (err) {
		await rm(part, { force: true })
		throw err
	}
	await rename(part, dest)
}

function openZip(path: string): Promise<yauzl.ZipFile> {
	return new Promise((resolve, reject) =>
		yauzl.open(path, { lazyEntries: true, autoClose: false }, (err, zip) =>
			err || !zip ? reject(err ?? new Error(`Cannot open ${path}`)) : resolve(zip),
		),
	)
}

/** Extract and gunzip selected .gz members of a zip. members: path inside zip → destination file. */
async function extractGzipMembers(zipPath: string, members: Map<string, string>): Promise<void> {
	const zip = await openZip(zipPath)
	const remaining = new Set(members.keys())
	try {
		await new Promise<void>((resolve, reject) => {
			zip.on("error", reject)
			zip.on("end", () =>
				remaining.size ? reject(new Error(`${zipPath} is missing: ${[...remaining].join(", ")}`)) : resolve(),
			)
			zip.on("entry", (entry: yauzl.Entry) => {
				const dest = members.get(entry.fileName)
				if (!dest) {
					zip.readEntry()
					return
				}
				zip.openReadStream(entry, (err, stream) => {
					if (err || !stream) {
						reject(err ?? new Error(`Cannot read ${entry.fileName}`))
						return
					}
					const part = `${dest}.part`
					pipeline(stream, createGunzip(), createWriteStream(part))
						.then(() => rename(part, dest))
						.then(() => {
							console.log(`  extracted ${entry.fileName}`)
							remaining.delete(entry.fileName)
							if (remaining.size === 0) resolve()
							else zip.readEntry()
						})
						.catch(async (e: unknown) => {
							await rm(part, { force: true })
							reject(e)
						})
				})
			})
			zip.readEntry()
		})
	} finally {
		zip.close()
	}
}

function offlineHelp(spec: DatasetSpec, paths: Record<SplitFile, string>, missing: SplitFile[]): string {
	const lines = [`--offline: missing dataset files for ${spec.id}:`, ...missing.map((k) => `  ${paths[k]}`)]
	const src = spec.source
	if (src.kind === "gzip-files") {
		lines.push("Download and gunzip:", ...missing.map((k) => `  ${src.urls[k]} → ${paths[k]}`))
	} else {
		lines.push(
			`Download ${src.url}, then gunzip these archive members:`,
			...missing.map((k) => `  ${src.members[k]} → ${paths[k]}`),
		)
	}
	return lines.join("\n")
}

/** Make sure the four decompressed IDX files exist in dataDir/<id>/ and return their paths. */
export async function ensureDatasetFiles(spec: DatasetSpec, opts: EnsureOptions): Promise<Record<SplitFile, string>> {
	const dir = join(opts.dataDir, spec.id)
	const paths = Object.fromEntries(SPLIT_FILES.map((k) => [k, join(dir, `${k}.idx`)])) as Record<SplitFile, string>
	const missing = SPLIT_FILES.filter((k) => !existsSync(paths[k]))
	if (missing.length === 0) return paths
	if (opts.offline) throw new Error(offlineHelp(spec, paths, missing))

	await mkdir(dir, { recursive: true })
	const src = spec.source
	if (src.kind === "gzip-files") {
		for (const k of missing) {
			const gz = `${paths[k]}.gz`
			console.log(`Downloading ${src.urls[k]}`)
			await downloadFile(src.urls[k], gz)
			await gunzipFile(gz, paths[k])
			await rm(gz)
		}
	} else {
		const archive = join(opts.dataDir, "_downloads", src.archiveName)
		if (!existsSync(archive)) {
			console.log(`Downloading ${src.url} (one-time, ~560 MB)`)
			await downloadFile(src.url, archive)
		}
		console.log(`Extracting ${spec.id} from ${archive}`)
		await extractGzipMembers(archive, new Map(missing.map((k) => [src.members[k], paths[k]])))
		if (opts.cleanDownloads) await rm(archive)
	}
	return paths
}
