import { mulberry32 } from "../nn/rng"
import { displayChar } from "../nn/transformer/tokenizer"
import type { ProbabilityChart } from "./chart"
import { byId, el } from "./dom"
import { isTextModel, type LoadedModel } from "./model-store"
import { FAST_MS, Player, playbackUnit, STEPS, speedToDelay } from "./player"
import type { SceneHost } from "./scene-host"
import type { TextScene } from "./text-scene"
import { TextSession } from "./text-session"
import {
	buildTextViewModel,
	caption,
	captionParts,
	mlpWeightsOf,
	type TextViewModel,
	VISIBLE_LETTERS,
} from "./text-view-model"
import type { ViewMode } from "./view-mode"

const TOP_CANDIDATES = 10
const PICK_STEP = STEPS.length - 1

/** Text box, step player, captions, next-character chart and the transformer's 3D network. */
export class TextMode implements ViewMode {
	private readonly panelEl = byId("textPanel")
	private readonly captionEl = byId("caption")
	private readonly prompt = byId<HTMLTextAreaElement>("prompt")
	private readonly ignored = byId("ignored")
	private readonly playBtn = byId<HTMLButtonElement>("playBtn")
	private readonly pills = byId("stepPills")
	private readonly speed = byId<HTMLInputElement>("speed")
	private readonly temperature = byId<HTMLInputElement>("temperature")
	private readonly headSelect = byId<HTMLSelectElement>("headSelect")
	private readonly follow = byId<HTMLInputElement>("followCamera")
	private readonly rand = mulberry32(Date.now() >>> 0)
	private readonly player = new Player()
	/** The current model; a new one per model, so nothing carries over between models. */
	private session: TextSession | null = null
	/** Vocabulary with whitespace made visible. */
	private labels: string[] = []
	/** Clicked character (window position), or null for the newest one. */
	private selected: number | null = null
	private vm: TextViewModel | null = null
	/** The character drawn at step ⑦ for the current prediction: raw text and its visible label. */
	private pick: { char: string; label: string; p: number } | null = null
	private timer: ReturnType<typeof setTimeout> | null = null
	/** True while walking through the steps (one row highlighted); otherwise the whole network is lit. */
	private stepping = false

	constructor(
		private readonly scene: TextScene,
		private readonly host: SceneHost,
		private readonly chart: ProbabilityChart,
	) {
		this.prompt.addEventListener("input", () => this.restart())
		byId("textClearBtn").addEventListener("click", () => {
			this.prompt.value = ""
			this.restart()
		})
		this.playBtn.addEventListener("click", () => (this.timer ? this.pause() : this.play()))
		byId("stepBtn").addEventListener("click", () => {
			this.pause()
			this.stepping = true
			this.forward()
		})
		byId("backBtn").addEventListener("click", () => {
			this.pause()
			this.stepping = true
			this.player.back()
			this.showStep()
		})
		byId("nextLetterBtn").addEventListener("click", () => {
			this.pause()
			this.writeLetter()
		})
		this.speed.addEventListener("input", () => this.renderSpeed())
		this.temperature.addEventListener("input", () => {
			byId("temperatureValue").textContent = Number(this.temperature.value).toFixed(2)
			this.renderCaption()
		})
		this.headSelect.addEventListener("change", () => this.rebuild())
		this.scene.onSelect = (letter) => {
			if (!this.vm) return
			this.pause()
			this.selected = this.vm.first + letter
			this.pick = null
			this.stepping = false
			this.rebuild()
			this.showStep() // fly to the newly selected letter
		}
		STEPS.forEach((name, i) => {
			const pill = el("button", undefined, name)
			pill.type = "button"
			pill.addEventListener("click", () => {
				this.pause()
				this.stepping = true
				this.player.jumpTo(i)
				this.showStep()
			})
			this.pills.append(pill)
		})
		this.renderSpeed()
	}

	enter(model: LoadedModel): void {
		if (!isTextModel(model)) throw new Error("TextMode can only show text models")
		this.vm = null // before pause(), which redraws the caption from the view model
		this.pause()
		this.session = new TextSession(model.info)
		this.labels = model.info.labels.map(displayChar)
		this.selected = null
		this.pick = null
		this.vm = null
		this.stepping = false
		this.player.reset()
		this.panelEl.classList.remove("hidden")
		byId("textTitle").textContent = `Text · ${model.info.title}`
		const all = el("option", undefined, "All heads")
		all.value = ""
		this.headSelect.replaceChildren(all)
		for (let h = 0; h < this.session.net.config.heads; h++) {
			const option = el("option", undefined, `Head ${h + 1} only`)
			option.value = String(h)
			this.headSelect.append(option)
		}
		this.chart.setLabels([])
		this.host.show(this.scene)
		this.host.frame(this.scene.cameraPosition(), this.scene.cameraTarget())
	}

	setWeights(params: Float32Array[]): void {
		if (!this.session) return
		this.pause() // a different snapshot is a different model: stop writing with it
		this.session.setWeights(params)
		this.pick = null
		this.rebuild()
	}

	leave(): void {
		this.vm = null
		this.pause()
		this.session = null
		this.panelEl.classList.add("hidden")
		this.captionEl.classList.add("hidden")
	}

	private play(): void {
		if (!this.session) return
		if (this.selected !== null) {
			// Playing writes text, which continues after the newest character.
			this.selected = null
			this.pick = null
			this.rebuild()
		}
		this.playBtn.textContent = "⏸ Pause"
		this.schedule()
	}

	private pause(): void {
		if (this.timer) clearTimeout(this.timer)
		this.timer = null
		this.playBtn.textContent = "▶ Play"
		this.renderCaption() // fast playback hides the caption; paused, it's readable again
	}

	private schedule(): void {
		this.timer = setTimeout(() => {
			if (playbackUnit(this.delay()) === "letter") this.writeLetter()
			else {
				this.stepping = true
				this.forward()
			}
			if (this.timer) this.schedule()
		}, this.delay())
	}

	/** Finish the current letter at once: pick it (step ⑦) and add it to the text. */
	private writeLetter(): void {
		this.stepping = false
		this.player.jumpTo(PICK_STEP)
		this.showStep(false)
		this.forward()
	}

	private delay(): number {
		return speedToDelay(Number(this.speed.value))
	}

	/** One step forward; leaving ⑦ writes the picked character (after the newest one) and starts over. */
	private forward(): void {
		if (!this.vm) return
		if (this.player.forward() === "append") {
			if (this.vm.isNewest && this.pick && this.session) {
				this.prompt.value = this.session.continueText(this.prompt.value, this.pick.char)
				this.prompt.scrollTop = this.prompt.scrollHeight
			}
			this.selected = null
			this.pick = null
			this.rebuild()
		}
		this.showStep()
	}

	/** Text changed: stop, and start the prediction for the new last character from ①. */
	private restart(): void {
		this.pause()
		this.stepping = false
		this.player.reset()
		this.selected = null
		this.pick = null
		this.rebuild()
		this.showStep()
	}

	/** New forward pass → new view model → rebuilt 3D network and chart. */
	private rebuild(): void {
		const session = this.session
		const read = session?.read(this.prompt.value)
		if (!session || !read) return
		this.ignored.textContent = read.ignored.length
			? `Not in this model's alphabet, ignored: ${read.ignored.map(displayChar).join(" ")}`
			: ""
		const newest = read.ids.length - 1
		const first = Math.max(0, newest + 1 - VISIBLE_LETTERS)
		if (this.selected !== null && (this.selected > newest || this.selected < first)) this.selected = null
		const head = this.headSelect.value === "" ? null : Number(this.headSelect.value)
		const config = session.net.config
		this.vm = buildTextViewModel(
			read.fwd,
			config,
			this.labels,
			this.selected ?? newest,
			head,
			mlpWeightsOf(session.net),
		)
		this.scene.update(this.vm)
		const sel = this.vm.first + this.vm.selected
		const V = config.vocab
		this.chart.updateTop(this.labels, read.fwd.probs.subarray(sel * V, (sel + 1) * V), Math.min(TOP_CANDIDATES, V))
		this.showStep(false)
	}

	/** Light the current step, draw the pick at ⑦, update pills and caption, move the camera. */
	private showStep(fly = true): void {
		const vm = this.vm
		const session = this.session
		if (!vm || !session) return
		const step = this.player.step
		if (step === PICK_STEP && !this.pick) {
			const drawn = session.pickAt(vm.first + vm.selected, Number(this.temperature.value), this.rand)
			this.pick = drawn ? { char: drawn.char, label: this.labels[drawn.id], p: drawn.p } : null
		}
		this.scene.showPick(step === PICK_STEP ? (this.pick?.label ?? null) : null)
		const fast = this.timer !== null && this.delay() < FAST_MS
		const flyMs = fly && this.follow.checked && !fast ? Math.min(900, this.delay() * 0.6) : null
		this.scene.setStep(step, this.stepping, flyMs)
		this.pills.querySelectorAll("button").forEach((pill, i) => {
			pill.className = !this.stepping ? "" : i < step ? "done" : i === step ? "on" : ""
		})
		this.renderCaption()
	}

	private renderCaption(): void {
		const vm = this.vm
		if (!vm) return
		const fast = this.timer !== null && this.delay() < FAST_MS
		this.captionEl.classList.toggle("hidden", fast)
		if (fast) return
		const pick = this.pick ? { char: this.pick.label, p: this.pick.p } : null
		const text = caption(this.player.step, vm, pick, Number(this.temperature.value))
		this.captionEl.replaceChildren(...captionParts(text).map((p) => (p.bold ? el("b", undefined, p.text) : p.text)))
	}

	private renderSpeed(): void {
		const ms = this.delay()
		byId("speedValue").textContent =
			playbackUnit(ms) === "letter"
				? `${Math.round(1000 / ms)} letters/s`
				: ms >= 1000
					? `${(ms / 1000).toFixed(1)} s per step`
					: `${Math.round(ms)} ms per step`
	}
}
