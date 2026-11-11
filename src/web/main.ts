import "./styles.css"
import type { ModelIndexEntry } from "../nn/model-format"
import { ProbabilityChart } from "./chart"
import { byId, el } from "./dom"
import { ImageMode } from "./image-mode"
import { isTextModel, type LoadedModel, ModelStore } from "./model-store"
import { NeuronPanel } from "./neuron-panel"
import { NetworkScene } from "./scene"
import { SceneHost } from "./scene-host"
import { loadFirstWorking } from "./startup"
import { TextMode } from "./text-mode"
import { TextScene } from "./text-scene"
import { describeImageSnapshot, describeTextSnapshot, TimelineControl, type TimelineEntry } from "./timeline"
import type { ViewMode } from "./view-mode"

const LAST_MODEL_KEY = "nn-vis:last-model"

function showFatal(message: string): void {
	const fatal = byId("fatal")
	fatal.textContent = message
	fatal.classList.remove("hidden")
}

function showBanner(message: string | null): void {
	const banner = byId("banner")
	banner.textContent = message ?? ""
	banner.classList.toggle("hidden", !message)
}

function entryLabel(entry: ModelIndexEntry): string {
	return `${entry.title} · ${entry.arch.toUpperCase()} (${(entry.finalAccuracy * 100).toFixed(1)}%)`
}

function timelineEntries(model: LoadedModel): TimelineEntry[] {
	const { info } = model
	if (info.arch === "transformer") {
		return info.snapshots.map((s, i, all) => ({ label: describeTextSnapshot(s, i, all.length), sample: s.sample }))
	}
	return info.snapshots.map((s, i, all) => ({ label: describeImageSnapshot(s, i, all.length), sample: null }))
}

async function main(): Promise<void> {
	const store = new ModelStore()
	const entries = await store.listModels()
	if (entries.length === 0) {
		showFatal("No trained models yet.\nRun:  npm run train -- --dataset mnist\nthen reload this page.")
		return
	}

	const host = new SceneHost(byId<HTMLCanvasElement>("scene"))
	const chart = new ProbabilityChart(byId("chart"))
	const panel = new NeuronPanel(byId("neuronPanel"), () => imageMode.clearSelection())
	const imageMode = new ImageMode(new NetworkScene(), host, chart, panel)
	const textMode = new TextMode(new TextScene(host), host, chart)

	let mode: ViewMode | null = null
	let model: LoadedModel | null = null
	let snapshotRequest = 0
	let modelRequest = 0
	/** Set once the user picks a model, so startup stops choosing one for them. */
	let userPicked = false

	const timeline = new TimelineControl(
		byId<HTMLInputElement>("timelineSlider"),
		byId("timelineLabel"),
		byId("timelineSample"),
		byId("timelineError"),
		(i) => {
			void selectSnapshot(i)
		},
	)

	async function selectSnapshot(index: number): Promise<void> {
		const current = model
		const currentMode = mode
		if (!current || !currentMode) return
		const request = ++snapshotRequest
		try {
			const params = await store.loadSnapshot(current, index)
			if (request !== snapshotRequest || current !== model) return // a newer request won
			timeline.showError(null)
			currentMode.setWeights(params)
		} catch (err) {
			if (request === snapshotRequest) timeline.showError(`Could not load snapshot: ${(err as Error).message}`)
		}
	}

	async function selectModel(entry: ModelIndexEntry): Promise<void> {
		const request = ++modelRequest
		const loaded = await store.loadModel(entry)
		if (request !== modelRequest) return // the user picked another model meanwhile
		select.value = entry.id
		const next: ViewMode = isTextModel(loaded) ? textMode : imageMode
		// Every other mode leaves, not just the current one: at startup no mode may have been entered yet.
		for (const other of [imageMode, textMode]) if (other !== next) other.leave()
		next.enter(loaded)
		mode = next
		model = loaded
		showBanner(null)
		const last = loaded.info.snapshots.length - 1
		timeline.setSnapshots(timelineEntries(loaded), last)
		await selectSnapshot(last)
	}

	const select = byId<HTMLSelectElement>("modelSelect")
	for (const entry of entries) {
		const option = el("option", undefined, entryLabel(entry))
		option.value = entry.id
		select.append(option)
	}
	select.addEventListener("change", () => {
		const entry = entries.find((e) => e.id === select.value)
		if (!entry) return
		userPicked = true
		try {
			localStorage.setItem(LAST_MODEL_KEY, entry.id)
		} catch {
			// storage unavailable: just don't remember the choice
		}
		selectModel(entry).catch((err: unknown) => showBanner(`Could not load ${entry.title}: ${(err as Error).message}`))
	})

	let remembered: string | null = null
	try {
		remembered = localStorage.getItem(LAST_MODEL_KEY)
	} catch {
		remembered = null
	}
	// MLP ids equal the dataset id, so a choice remembered before CNN support still matches.
	const initial = entries.find((e) => e.id === remembered) ?? entries[0]
	const ordered = [initial, ...entries.filter((e) => e !== initial)]
	const { loaded, failures } = await loadFirstWorking(ordered, selectModel, () => !userPicked)
	const problems = failures.map((f) => `Could not load ${f.entry.title}: ${f.error.message}`).join("\n")
	if (!loaded && !userPicked) {
		showFatal(`${problems}\n\nRetrain with:  npm run train -- --dataset <name>`)
		return
	}
	if (problems) showBanner(problems)
}

main().catch((err: unknown) => showFatal(`Failed to start: ${(err as Error).message}`))
