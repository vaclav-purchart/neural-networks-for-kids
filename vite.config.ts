import { fileURLToPath } from "node:url"
import { defineConfig } from "vite"

const fromRoot = (p: string) => fileURLToPath(new URL(p, import.meta.url))

export default defineConfig({
	root: fromRoot("./src/web"),
	publicDir: fromRoot("./public"),
	// three.js alone is ~700 kB; one chunk is fine for a local demo.
	build: { outDir: fromRoot("./dist"), emptyOutDir: true, chunkSizeWarningLimit: 1500 },
})
