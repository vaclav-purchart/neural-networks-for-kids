import type { SnapshotMeta, TextSnapshotMeta } from "../nn/model-format"

export interface TimelineEntry {
	label: string
	/** Generated text shown under the label (text models), or null. */
	sample: string | null
}

export function describeImageSnapshot(s: SnapshotMeta, i: number, n: number): string {
	const parts = [
		`Snapshot ${i + 1}/${n}`,
		`${s.imagesSeen.toLocaleString("en-US")} images seen (${s.epoch.toFixed(2)} epochs)`,
		`test accuracy ${(s.testAccuracy * 100).toFixed(1)}% on ${s.testSamples.toLocaleString("en-US")} images`,
	]
	if (s.avgLoss !== null) parts.push(`loss ${s.avgLoss.toFixed(3)}`)
	return parts.join(" · ")
}

export function describeTextSnapshot(s: TextSnapshotMeta, i: number, n: number): string {
	return [
		`Snapshot ${i + 1}/${n}`,
		`step ${s.steps.toLocaleString("en-US")}`,
		`${s.charsSeen.toLocaleString("en-US")} characters seen`,
		`test loss ${s.testLoss.toFixed(3)}`,
		`next-character accuracy ${(s.testAccuracy * 100).toFixed(1)}%`,
	].join(" · ")
}

/** Slider over training snapshots. */
export class TimelineControl {
	private entries: readonly TimelineEntry[] = []
	private readonly slider: HTMLInputElement
	private readonly label: HTMLElement
	private readonly sample: HTMLElement
	private readonly error: HTMLElement

	constructor(
		slider: HTMLInputElement,
		label: HTMLElement,
		sample: HTMLElement,
		error: HTMLElement,
		onChange: (index: number) => void,
	) {
		this.slider = slider
		this.label = label
		this.sample = sample
		this.error = error
		slider.addEventListener("input", () => {
			this.render()
			onChange(Number(slider.value))
		})
	}

	setSnapshots(entries: readonly TimelineEntry[], selected: number): void {
		this.entries = entries
		this.slider.min = "0"
		this.slider.max = String(Math.max(0, entries.length - 1))
		this.slider.value = String(selected)
		this.slider.disabled = entries.length < 2
		this.showError(null)
		this.render()
	}

	showError(message: string | null): void {
		this.error.textContent = message ?? ""
	}

	private render(): void {
		const entry = this.entries[Number(this.slider.value)]
		this.label.textContent = entry?.label ?? ""
		this.sample.textContent = entry?.sample ?? ""
		this.sample.classList.toggle("hidden", !entry?.sample)
	}
}
