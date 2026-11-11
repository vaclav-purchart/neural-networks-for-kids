import { el } from "./dom"

/** One button per class that loads a random test image of that class into the sketchpad. */
export class SampleButtons {
	private readonly host: HTMLElement
	private readonly onPick: (pixels: Float32Array) => void

	constructor(host: HTMLElement, onPick: (pixels: Float32Array) => void) {
		this.host = host
		this.onPick = onPick
	}

	setModel(labels: readonly string[], samples: Uint8Array | null, perClass: number, imageSize: number): void {
		this.host.replaceChildren()
		if (!samples || perClass === 0) {
			this.host.append(el("span", "muted", "No sample images in this model."))
			return
		}
		this.host.append(el("span", "muted", "Load a random test image:"))
		const grid = el("div", "sample-grid")
		labels.forEach((label, c) => {
			const button = el("button", "sample-btn", label)
			button.type = "button"
			button.title = `Random test image of "${label}"`
			button.addEventListener("click", () => {
				const offset = (c * perClass + Math.floor(Math.random() * perClass)) * imageSize
				const pixels = new Float32Array(imageSize)
				for (let i = 0; i < imageSize; i++) pixels[i] = samples[offset + i] / 255
				this.onPick(pixels)
			})
			grid.append(button)
		})
		this.host.append(grid)
	}
}
