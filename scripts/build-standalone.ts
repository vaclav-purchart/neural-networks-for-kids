/**
 * Builds dist/nn-vis-standalone.html: one file with the app, its styles and every trained model
 * baked in. It opens straight from disk (file://), with no server and no network access.
 */
import { mkdir, mkdtemp, readdir, readFile, rm, stat, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join, relative, sep } from "node:path"
import { fileURLToPath } from "node:url"
import { build } from "vite"
import type { EmbeddedFiles } from "../src/web/model-source"

const root = fileURLToPath(new URL("..", import.meta.url))
const modelsDir = join(root, "public", "models")
const outFile = join(root, "dist", "nn-vis-standalone.html")

async function listFiles(dir: string): Promise<string[]> {
	const entries = await readdir(dir, { withFileTypes: true, recursive: true })
	return entries.filter((e) => e.isFile() && e.name !== ".DS_Store").map((e) => join(e.parentPath, e.name))
}

async function embedModels(): Promise<EmbeddedFiles> {
	const files: EmbeddedFiles = {}
	for (const file of (await listFiles(modelsDir)).sort()) {
		const path = relative(modelsDir, file).split(sep).join("/")
		const data = await readFile(file)
		files[path] = path.endsWith(".json") ? data.toString("utf8") : { bin: data.toString("base64") }
	}
	if (!files["index.json"]) throw new Error("No trained models in public/models. Run: npm run train -- --dataset mnist")
	return files
}

/** Keeps a "</script" inside inlined code from ending the script element early. */
const scriptSafe = (code: string) => code.replace(/<\/(script)/gi, "<\\/$1")

async function main(): Promise<void> {
	const models = await embedModels()
	const tmp = await mkdtemp(join(tmpdir(), "nn-vis-standalone-"))
	try {
		await build({
			configFile: join(root, "vite.config.ts"),
			base: "./",
			publicDir: false,
			logLevel: "warn",
			build: { outDir: tmp, emptyOutDir: true, modulePreload: false, assetsInlineLimit: Number.MAX_SAFE_INTEGER },
		})
		const built = await listFiles(tmp)
		const js = built.filter((f) => f.endsWith(".js"))
		const css = built.filter((f) => f.endsWith(".css"))
		const other = built.filter((f) => !/\.(js|css|html)$/.test(f))
		if (js.length !== 1 || other.length > 0) throw new Error(`Expected one script and no other assets, got: ${built}`)

		let html = await readFile(join(tmp, "index.html"), "utf8")
		html = html.replace(/<script type="module"[^>]*src="[^"]+"[^>]*><\/script>\s*/, "")
		html = html.replace(/<link rel="stylesheet"[^>]*>\s*/g, "")
		const styles = await Promise.all(css.map((f) => readFile(f, "utf8")))
		const app = await readFile(js[0], "utf8")
		const data = JSON.stringify(models).replace(/</g, "\\u003c")
		html = html.replace("</head>", () => `<style>\n${styles.join("\n")}</style>\n</head>`)
		html = html.replace(
			"</body>",
			() =>
				`<script>globalThis.__NN_VIS_MODELS__ = ${data}</script>\n` +
				`<script type="module">\n${scriptSafe(app)}</script>\n</body>`,
		)
		await mkdir(join(root, "dist"), { recursive: true })
		await writeFile(outFile, html)
		const mb = (await stat(outFile)).size / 1e6
		console.log(`Wrote ${relative(root, outFile)} (${mb.toFixed(1)} MB, ${Object.keys(models).length} model files)`)
	} finally {
		await rm(tmp, { recursive: true, force: true })
	}
}

main().catch((err: unknown) => {
	console.error((err as Error).message)
	process.exit(1)
})
