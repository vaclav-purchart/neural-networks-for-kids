import { el } from "./dom"

export interface ViewSettings {
	connectionsPerNeuron: number
	minWeight: number
	brushRadius: number
	/** Scale drawings to the dataset's character size before predicting. */
	normalizeSize: boolean
}

export const DEFAULT_SETTINGS: ViewSettings = {
	connectionsPerNeuron: 16,
	minWeight: 0,
	brushRadius: 1.4,
	normalizeSize: true,
}

interface SliderDef {
	key: Exclude<keyof ViewSettings, "normalizeSize">
	label: string
	hint: string
	min: number
	max: number
	step: number
	format: (v: number) => string
}

const SLIDERS: SliderDef[] = [
	{
		key: "connectionsPerNeuron",
		label: "Connections per neuron",
		hint: "Strongest incoming weights drawn for each neuron.",
		min: 1,
		max: 64,
		step: 1,
		format: String,
	},
	{
		key: "minWeight",
		label: "Hide weights below",
		hint: "Connections with a smaller |weight| are not drawn.",
		min: 0,
		max: 0.5,
		step: 0.01,
		format: (v) => v.toFixed(2),
	},
	{
		key: "brushRadius",
		label: "Brush size",
		hint: "Radius of the drawing brush in grid cells.",
		min: 0.6,
		max: 3,
		step: 0.1,
		format: (v) => v.toFixed(1),
	},
]

export class SettingsPanel {
	constructor(
		panel: HTMLElement,
		toggleButton: HTMLButtonElement,
		initial: ViewSettings,
		onChange: (settings: ViewSettings, changed: keyof ViewSettings) => void,
	) {
		const values = { ...initial }
		toggleButton.addEventListener("click", () => panel.classList.toggle("hidden"))
		panel.append(el("h2", undefined, "Settings"))
		for (const def of SLIDERS) {
			const input = el("input")
			input.type = "range"
			input.min = String(def.min)
			input.max = String(def.max)
			input.step = String(def.step)
			input.value = String(values[def.key])
			const output = el("span", "muted", def.format(values[def.key]))
			input.addEventListener("input", () => {
				values[def.key] = Number(input.value)
				output.textContent = def.format(values[def.key])
				onChange({ ...values }, def.key)
			})
			const head = el("div", "setting-head")
			head.append(el("span", undefined, def.label), output)
			const wrap = el("label", "setting")
			wrap.append(head, input, el("div", "muted", def.hint))
			panel.append(wrap)
		}

		const box = el("input")
		box.type = "checkbox"
		box.checked = values.normalizeSize
		box.addEventListener("change", () => {
			values.normalizeSize = box.checked
			onChange({ ...values }, "normalizeSize")
		})
		const toggle = el("label", "setting setting-toggle")
		toggle.append(box, el("span", undefined, " Normalise drawing size"))
		toggle.append(el("div", "muted", "Scale your drawing to the dataset's character size before predicting."))
		panel.append(toggle)
	}
}
