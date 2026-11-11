import { el } from "./dom"

/** Indices of the k largest values, largest first. */
export function topK(values: ArrayLike<number>, k: number): number[] {
	return Array.from({ length: values.length }, (_, i) => i)
		.sort((a, b) => values[b] - values[a])
		.slice(0, k)
}

/** One horizontal bar per class; the most likely class is highlighted. */
export class ProbabilityChart {
	private readonly host: HTMLElement
	private rows: { row: HTMLElement; label: HTMLElement; bar: HTMLElement; value: HTMLElement }[] = []

	constructor(host: HTMLElement) {
		this.host = host
	}

	setLabels(labels: readonly string[]): void {
		this.host.replaceChildren()
		this.host.classList.toggle("compact", labels.length > 12)
		this.rows = labels.map((text) => {
			const row = el("div", "chart-row")
			const label = el("span", "chart-label", text)
			const track = el("div", "chart-track")
			const bar = el("div", "chart-bar")
			const value = el("span", "chart-value", "0.0%")
			track.append(bar)
			row.append(label, track, value)
			this.host.append(row)
			return { row, label, bar, value }
		})
	}

	update(probs: ArrayLike<number>): void {
		let best = 0
		for (let i = 1; i < this.rows.length; i++) if (probs[i] > probs[best]) best = i
		this.rows.forEach((r, i) => {
			const pct = `${(probs[i] * 100).toFixed(1)}%`
			r.bar.style.width = pct
			r.value.textContent = pct
			r.row.classList.toggle("top", i === best)
		})
	}

	/** Show only the k most likely labels, most likely first (for large vocabularies). */
	updateTop(labels: readonly string[], probs: ArrayLike<number>, k: number): void {
		const order = topK(probs, k)
		if (this.rows.length !== order.length) this.setLabels(order.map(() => ""))
		order.forEach((id, i) => {
			const r = this.rows[i]
			const pct = `${(probs[id] * 100).toFixed(1)}%`
			r.label.textContent = labels[id]
			r.bar.style.width = pct
			r.value.textContent = pct
			r.row.classList.toggle("top", i === 0)
		})
	}
}
