import { buildNetwork } from "../nn/architectures"
import { inspectFeatureMap, inspectNeuron } from "../nn/inspect"
import type { Network } from "../nn/network"
import { normalize, prepareDrawing } from "../nn/preprocess"
import type { ProbabilityChart } from "./chart"
import { byId } from "./dom"
import { isImageModel, type LoadedImageModel, type LoadedModel } from "./model-store"
import type { NeuronPanel, NeuronRef } from "./neuron-panel"
import { SampleButtons } from "./samples"
import type { NetworkScene } from "./scene"
import type { SceneHost } from "./scene-host"
import { DEFAULT_SETTINGS, SettingsPanel, type ViewSettings } from "./settings"
import { Sketchpad } from "./sketchpad"
import type { ViewMode } from "./view-mode"

/** Display name of act[layer]: "Input layer", "Conv 1", "Hidden layer 2", "Output layer". */
function layerName(net: Network, layer: number): string {
	if (layer === 0) return "Input layer"
	if (layer === net.layers.length) return "Output layer"
	const kind = net.layers[layer - 1].kind
	const n = net.layers.slice(0, layer).filter((l) => l.kind === kind).length
	if (kind === "conv2d") return `Conv ${n}`
	if (kind === "maxpool") return `Pool ${n}`
	return `Hidden layer ${n}`
}

/** Name of one value in act[layer]: a pixel, "#12", "map 3 · (2, 4)" or a class label. */
function valueName(model: LoadedImageModel, net: Network, layer: number, index: number): string {
	const { info } = model
	if (layer === 0) {
		const cols = info.inputShape[1]
		return `pixel (${Math.floor(index / cols)}, ${index % cols})`
	}
	if (layer === net.layers.length) return `"${info.labels[index]}"`
	const l = net.layers[layer - 1]
	if (l.kind === "dense") return `#${index + 1}`
	const { h, w } = l.outShape
	const plane = h * w
	return `map ${Math.floor(index / plane) + 1} · (${Math.floor((index % plane) / w)}, ${index % w})`
}

/** Drawing pad, sample buttons and the layer-by-layer 3D view of an MLP or CNN. */
export class ImageMode implements ViewMode {
	private readonly sketch = new Sketchpad(byId("sketchpad"))
	private readonly samples: SampleButtons
	private readonly panelEl = byId("imagePanel")
	private settings: ViewSettings = { ...DEFAULT_SETTINGS }
	private model: LoadedImageModel | null = null
	private net: Network | null = null
	private weightsLoaded = false
	private selected: NeuronRef | null = null

	constructor(
		private readonly scene: NetworkScene,
		private readonly host: SceneHost,
		private readonly chart: ProbabilityChart,
		private readonly panel: NeuronPanel,
	) {
		this.sketch.brushRadius = this.settings.brushRadius
		this.samples = new SampleButtons(byId("samples"), (pixels) => this.sketch.setPixels(pixels))
		new SettingsPanel(byId("settingsPanel"), byId<HTMLButtonElement>("settingsBtn"), this.settings, (next, changed) => {
			this.settings = next
			if (changed === "brushRadius") {
				this.sketch.brushRadius = next.brushRadius
			} else if (changed === "normalizeSize") {
				this.refresh()
			} else if (this.net && this.weightsLoaded) {
				this.scene.setWeights(this.net, this.connectionOptions())
				this.refresh()
			}
		})
		byId("clearBtn").addEventListener("click", () => this.sketch.clear())
		this.sketch.onChange = () => this.refresh()
		this.scene.onNeuronClick = (ref) => {
			this.selected = ref
			this.refresh()
		}
	}

	/** The neuron panel's close button. */
	clearSelection(): void {
		this.selected = null
		this.scene.highlight(null)
		this.refresh()
	}

	enter(model: LoadedModel): void {
		if (!isImageModel(model)) throw new Error("ImageMode cannot show a text model")
		const { info } = model
		const [rows, cols] = info.inputShape
		if (rows !== this.sketch.rows || cols !== this.sketch.cols) {
			throw new Error(
				`${info.title} expects ${rows}×${cols} images; this viewer draws ${this.sketch.rows}×${this.sketch.cols}`,
			)
		}
		this.model = model
		this.net = buildNetwork({ c: 1, h: rows, w: cols }, info.layers)
		this.weightsLoaded = false
		this.selected = null
		this.panel.hide()
		this.panelEl.classList.remove("hidden")
		byId("settingsBtn").classList.remove("hidden")
		byId("drawTitle").textContent = `Draw · ${info.title}`
		this.chart.setLabels(info.labels)
		this.samples.setModel(info.labels, model.samples, info.samples?.perClass ?? 0, rows * cols)
		this.scene.setModel(this.net, info.labels, info.inputShape)
		this.host.show(this.scene)
		this.host.frame(this.scene.cameraPosition())
	}

	setWeights(params: Float32Array[]): void {
		if (!this.net) return
		this.net.setParams(params)
		this.weightsLoaded = true
		this.scene.setWeights(this.net, this.connectionOptions())
		this.refresh()
	}

	leave(): void {
		this.model = null
		this.net = null
		this.selected = null
		this.panel.hide()
		this.panelEl.classList.add("hidden")
		byId("settingsBtn").classList.add("hidden")
		byId("settingsPanel").classList.add("hidden")
	}

	private connectionOptions() {
		return { perNeuron: this.settings.connectionsPerNeuron, minWeight: this.settings.minWeight }
	}

	private refresh(): void {
		const { model, net } = this
		if (!model || !net || !this.weightsLoaded) return
		const { info } = model
		const [rows, cols] = info.inputShape
		const centered = prepareDrawing(
			this.sketch.getPixels(),
			rows,
			cols,
			info.preprocessing,
			this.settings.normalizeSize,
		)
		const fwd = net.forward(normalize(centered, info.normalization.mean, info.normalization.std), 1)
		this.chart.update(fwd.act[fwd.act.length - 1])
		this.scene.update(fwd.act, centered)
		const selected = this.selected
		if (!selected) {
			this.panel.hide()
			return
		}
		if (net.layers[selected.layer - 1]?.kind === "conv2d") {
			const details = inspectFeatureMap(net, fwd, selected.layer, selected.index)
			this.panel.showFeatureMap(details, `${layerName(net, selected.layer)} · filter ${selected.index + 1}`)
			return
		}
		const nameOf = (layer: number, index: number) => valueName(model, net, layer, index)
		const title = `${layerName(net, selected.layer)} · ${nameOf(selected.layer, selected.index)}`
		this.panel.show(inspectNeuron(net, fwd, selected.layer, selected.index), title, nameOf)
	}
}
