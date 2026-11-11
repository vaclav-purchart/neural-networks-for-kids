/** Where model files come from, addressed by paths relative to the models folder ("mnist/model.json"). */
export interface ModelSource {
	/** File contents, or null when the file does not exist. */
	text(path: string): Promise<string | null>
	bytes(path: string): Promise<Uint8Array | null>
}

/** Files baked into the page: text as-is, binary files as { bin: base64 }. */
export type EmbeddedFiles = Record<string, string | { bin: string }>

declare global {
	/** Set by the standalone build (scripts/build-standalone.ts). */
	var __NN_VIS_MODELS__: EmbeddedFiles | undefined
}

// Accept: application/json stops Vite's dev server from answering missing files with index.html.
const JSON_HEADERS = { Accept: "application/json" }

/** Files served from the models/ folder next to the page. */
export function fetchSource(modelsUrl: string): ModelSource {
	async function get(path: string, init?: RequestInit): Promise<Response | null> {
		const res = await fetch(new URL(path, modelsUrl).href, init)
		if (res.status === 404) return null
		if (!res.ok) throw new Error(`models/${path}: HTTP ${res.status}`)
		return res
	}
	return {
		async text(path) {
			const res = await get(path, { cache: "no-store", headers: JSON_HEADERS })
			return res ? res.text() : null
		},
		async bytes(path) {
			const res = await get(path, { cache: "no-store" })
			return res ? new Uint8Array(await res.arrayBuffer()) : null
		},
	}
}

export function embeddedSource(files: EmbeddedFiles): ModelSource {
	const lookup = (path: string) => (Object.hasOwn(files, path) ? files[path] : undefined)
	return {
		async text(path) {
			const file = lookup(path)
			if (file === undefined) return null
			if (typeof file !== "string") throw new Error(`models/${path} is binary`)
			return file
		},
		async bytes(path) {
			const file = lookup(path)
			if (file === undefined) return null
			if (typeof file === "string") return new TextEncoder().encode(file)
			return Uint8Array.from(atob(file.bin), (c) => c.charCodeAt(0))
		},
	}
}

/** Embedded models when the page was built standalone, otherwise the models/ folder on the server. */
export function defaultSource(): ModelSource {
	if (globalThis.__NN_VIS_MODELS__) return embeddedSource(globalThis.__NN_VIS_MODELS__)
	return fetchSource(new URL(`${import.meta.env.BASE_URL}models/`, location.href).href)
}
