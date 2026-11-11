import type { Contribution, FeatureMapDetails, NeuronDetails } from "../nn/inspect"
import { el } from "./dom"

export interface NeuronRef {
	layer: number
	index: number
}

const fmt = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(3))

function signed(v: number): HTMLTableCellElement {
	return el("td", v > 0 ? "pos" : v < 0 ? "neg" : undefined, fmt(v))
}

function table(headers: string[], rows: HTMLTableCellElement[][]): HTMLTableElement {
	const t = el("table")
	const head = el("tr")
	for (const h of headers) head.append(el("th", undefined, h))
	t.append(head)
	for (const cells of rows) {
		const tr = el("tr")
		tr.append(...cells)
		t.append(tr)
	}
	return t
}

const MAX_KERNELS = 16

/** A 3×3 kernel as a tiny heat grid: warm = positive, cool = negative, stronger = larger |weight|. */
function kernelCanvas(weights: Float32Array, maxAbs: number): HTMLCanvasElement {
	const cell = 14
	const canvas = el("canvas", "kernel")
	canvas.width = cell * 3
	canvas.height = cell * 3
	const ctx = canvas.getContext("2d")
	if (!ctx) return canvas
	ctx.fillStyle = "#1c2536"
	ctx.fillRect(0, 0, canvas.width, canvas.height)
	for (let i = 0; i < 9; i++) {
		const w = weights[i]
		const alpha = Math.min(1, Math.abs(w) / maxAbs)
		ctx.fillStyle = w >= 0 ? `rgba(255, 159, 67, ${alpha})` : `rgba(61, 139, 253, ${alpha})`
		ctx.fillRect((i % 3) * cell, Math.floor(i / 3) * cell, cell - 1, cell - 1)
	}
	return canvas
}

/** Shows what a clicked neuron computes: bias + Σ inputs × weights → activation. */
export class NeuronPanel {
	private readonly host: HTMLElement
	private readonly onClose: () => void

	constructor(host: HTMLElement, onClose: () => void) {
		this.host = host
		this.onClose = onClose
	}

	show(d: NeuronDetails, title: string, nameOf: (layer: number, index: number) => string): void {
		this.host.replaceChildren()
		this.host.classList.remove("hidden")

		this.host.append(this.header(title))

		const summary = el("div")
		if (d.pre !== null && d.bias !== null) {
			summary.textContent = `bias ${fmt(d.bias)} + Σ(input × weight) = ${fmt(d.pre)} → activation ${fmt(d.activation)}`
		} else {
			summary.textContent = `pixel value after normalisation: ${fmt(d.activation)}`
		}
		this.host.append(summary)

		if (d.incoming.length) {
			this.host.append(el("h4", undefined, "Strongest incoming contributions"))
			this.host.append(
				table(
					["From", "Input", "Weight", "Input × weight"],
					d.incoming.map((c: Contribution) => [
						el("td", undefined, nameOf(d.layer - 1, c.index)),
						el("td", undefined, fmt(c.value)),
						signed(c.weight),
						signed(c.contribution),
					]),
				),
			)
		}
		if (d.outgoing.length) {
			this.host.append(el("h4", undefined, "Strongest outgoing contributions"))
			this.host.append(
				table(
					["To", "Weight", "Sent (act × weight)"],
					d.outgoing.map((c: Contribution) => [
						el("td", undefined, nameOf(d.layer + 1, c.index)),
						signed(c.weight),
						signed(c.contribution),
					]),
				),
			)
		}
	}

	/** What a conv filter looks for (its 3×3 weights per input channel) and where it fires. */
	showFeatureMap(d: FeatureMapDetails, title: string): void {
		this.host.replaceChildren()
		this.host.classList.remove("hidden")
		this.host.append(this.header(title))
		this.host.append(
			el(
				"div",
				undefined,
				`bias ${fmt(d.bias)} · strongest ${fmt(d.maxValue)} at (${d.maxAt[0]}, ${d.maxAt[1]}) · mean ${fmt(d.mean)} · ${d.height}×${d.width} map`,
			),
		)
		this.host.append(el("h4", undefined, `3×3 filter weights, one grid per input channel (${d.kernels.length})`))
		const shown = d.kernels.slice(0, MAX_KERNELS)
		let maxAbs = 1e-6
		for (const k of shown) for (const v of k) maxAbs = Math.max(maxAbs, Math.abs(v))
		const grid = el("div", "kernel-grid")
		shown.forEach((k, c) => {
			const item = el("div", "kernel-item")
			item.append(kernelCanvas(k, maxAbs), el("span", "muted", `in ${c + 1}`))
			grid.append(item)
		})
		this.host.append(grid)
		if (d.kernels.length > MAX_KERNELS) {
			this.host.append(el("div", "muted", `+${d.kernels.length - MAX_KERNELS} more input channels`))
		}
		this.host.append(el("div", "muted", "Warm = positive weight, cool = negative; stronger colour = larger weight."))
	}

	private header(title: string): HTMLElement {
		const close = el("button", undefined, "Close")
		close.type = "button"
		close.addEventListener("click", () => this.onClose())
		const header = el("div", "neuron-header")
		header.append(el("h3", undefined, title), close)
		return header
	}

	hide(): void {
		this.host.classList.add("hidden")
		this.host.replaceChildren()
	}
}
