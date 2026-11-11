import * as THREE from "three"
import { Line2 } from "three/addons/lines/Line2.js"
import { LineGeometry } from "three/addons/lines/LineGeometry.js"
import { LineMaterial } from "three/addons/lines/LineMaterial.js"
import { rowBrightness, STEPS } from "./player"
import { disposeObject, heat, type SceneContent, type SceneHost } from "./scene-host"
import { type TextViewModel, VISIBLE_LETTERS, type ViewArc } from "./text-view-model"

const LETTER_GAP = 2.4
const NEURON_GAP = 0.105
const HIDDEN_GAP = NEURON_GAP * 1.25
const NEURON_R = 0.055
/** Connection colors for positive / negative flow (same as the digit view). */
const POSITIVE = "#ff7a45"
const NEGATIVE = "#3fa7ff"
/** One color per attention head (cycled when there are more heads). */
export const HEAD_COLORS = ["#ff9f43", "#2ecc71", "#e056fd", "#3d8bfd", "#f8e71c", "#00d2d3", "#ff6b6b", "#a29bfe"]

/** Height of every row, bottom to top. */
const Y = { text: 0, embedding: 1.6, attention: 3.4, hidden: 4.7, block1: 6.0, block2: 8.2, output: 10.0, pick: 11.5 }
/** Row each player step lights up and frames. */
const STEP_Y = [Y.text, Y.embedding, Y.attention, Y.hidden, Y.block2, Y.output, Y.pick]

/** x of a letter slot; the newest letter is always in the rightmost slot. */
export function letterX(index: number, count: number): number {
	const slot = VISIBLE_LETTERS - count + index
	return (slot - (VISIBLE_LETTERS - 1) / 2) * LETTER_GAP
}

/** Points of a quadratic curve from `a` to `b`, bulging towards the camera by `bulge`. */
export function arcPoints(a: THREE.Vector3Like, b: THREE.Vector3Like, bulge: number, segments = 20): number[] {
	const cx = (a.x + b.x) / 2
	const cy = (a.y + b.y) / 2 + 0.3
	const cz = 2 * bulge
	const out: number[] = []
	for (let i = 0; i <= segments; i++) {
		const t = i / segments
		const p = (1 - t) * (1 - t)
		const q = 2 * (1 - t) * t
		const r = t * t
		out.push(p * a.x + q * cx + r * b.x, p * a.y + q * cy + r * b.y, p * a.z + q * cz + r * b.z)
	}
	return out
}

function spriteFrom(canvas: HTMLCanvasElement, w: number, h: number): THREE.Sprite {
	const texture = new THREE.CanvasTexture(canvas)
	texture.colorSpace = THREE.SRGBColorSpace
	const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, depthWrite: false }))
	sprite.scale.set(w, h, 1)
	return sprite
}

/** A row label: colored title, gray explanation underneath, left-aligned. */
function labelSprite(title: string, sub: string, color: string): THREE.Sprite {
	const canvas = document.createElement("canvas")
	canvas.width = 1024
	canvas.height = 160
	const ctx = canvas.getContext("2d")!
	ctx.textBaseline = "top"
	ctx.fillStyle = color
	ctx.font = "bold 58px system-ui, sans-serif"
	ctx.fillText(title, 0, 8)
	ctx.fillStyle = "#9aa4b6"
	ctx.font = "44px system-ui, sans-serif"
	ctx.fillText(sub, 0, 90)
	return spriteFrom(canvas, 6.4, 1)
}

/** Centered one-line text, `height` world units tall. */
function textSprite(text: string, color: string, height: number, bold = true): THREE.Sprite {
	const canvas = document.createElement("canvas")
	const ctx = canvas.getContext("2d")!
	const font = `${bold ? "bold " : ""}96px system-ui, sans-serif`
	ctx.font = font
	canvas.width = Math.max(128, Math.ceil(ctx.measureText(text).width) + 24)
	canvas.height = 128
	ctx.font = font // resizing the canvas resets the context
	ctx.fillStyle = color
	ctx.textBaseline = "middle"
	ctx.textAlign = "center"
	ctx.fillText(text, canvas.width / 2, 68)
	return spriteFrom(canvas, (height * canvas.width) / canvas.height, height)
}

function letterTile(char: string, selected: boolean): THREE.Mesh {
	const canvas = document.createElement("canvas")
	canvas.width = 128
	canvas.height = 128
	const ctx = canvas.getContext("2d")!
	ctx.fillStyle = selected ? "#3d8bfd" : "#2a3448"
	ctx.fillRect(0, 0, 128, 128)
	ctx.fillStyle = "#ffffff"
	ctx.font = "bold 84px ui-monospace, Menlo, monospace"
	ctx.textAlign = "center"
	ctx.textBaseline = "middle"
	ctx.fillText(char, 64, 70)
	const texture = new THREE.CanvasTexture(canvas)
	texture.colorSpace = THREE.SRGBColorSpace
	return new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ map: texture, transparent: true }))
}

/**
 * The transformer as a labeled network: rows ① text … ⑦ pick, bottom to top. Every letter has its own
 * column (its residual stream); attention arcs, the MLP and the output are drawn for the selected letter.
 */
export class TextScene implements SceneContent {
	readonly root = new THREE.Group()
	/** A clicked letter (index into the visible letters). */
	onSelect: ((letter: number) => void) | null = null
	private rows: THREE.Group[] = []
	private pickable: THREE.Object3D[] = []
	private vm: TextViewModel | null = null
	private step = 0
	private stepping = false
	private readonly color = new THREE.Color()
	private readonly matrix = new THREE.Matrix4()

	constructor(private readonly host: SceneHost) {}

	/** Rebuild everything for a new forward pass, selection or head filter. */
	update(vm: TextViewModel): void {
		this.clear()
		this.vm = vm
		const n = vm.letters.length
		const sx = letterX(vm.selected, n)
		const d = vm.config.dModel
		this.rows = STEPS.map(() => {
			const g = new THREE.Group()
			this.root.add(g)
			return g
		})
		const [rText, rEmb, rAtt, rMlp, rBlock2, rOut, rPick] = this.rows

		// Columns: each letter keeps its own vector (its residual stream) all the way up.
		const top = vm.blocks.length > 1 ? Y.block2 : Y.block1
		const guides: number[] = []
		vm.letters.forEach((_, i) => {
			guides.push(letterX(i, n), Y.text + 0.5, 0, letterX(i, n), top, 0)
		})
		const always = new THREE.Group()
		this.addSegments(always, guides, "#253048", 0.9)
		this.root.add(always)

		// ① text
		vm.letters.forEach((c, i) => {
			const tile = letterTile(c, i === vm.selected)
			tile.position.set(letterX(i, n), Y.text, 0)
			tile.userData.letter = i
			rText.add(tile)
			this.pickable.push(tile)
		})
		this.addLabel(rText, Y.text, "① Text", "the last characters the model reads", "#e5e9f2")

		// ② embedding: every letter's numbers
		this.addLetterRows(rEmb, vm.embedding, Y.embedding)
		this.addLabel(rEmb, Y.embedding, "② Embedding", `each character → ${d} numbers (16 shown)`, "#3d8bfd")
		if (vm.first > 0) this.addEarlierMarker(rEmb, Y.embedding)

		// ③ attention (block 1) into the selected letter
		this.addArcs(rAtt, vm.blocks[0].arcs, Y.embedding, Y.attention)
		this.addNeuronRow(rAtt, vm.blocks[0].afterAttention, sx, Y.attention, NEURON_GAP)
		this.addLabel(rAtt, Y.attention, "③ Attention", "looks back at earlier characters", "#ff9f43")

		// ④ MLP (block 1): the selected letter's in → hidden → out, plus every letter's block-1 output
		const hidden = vm.blocks[0].hidden.map((h) => h.value)
		const { inFlows, outFlows } = vm.blocks[0]
		this.addConnections(rMlp, inFlows, sx, Y.attention, NEURON_GAP, Y.hidden, HIDDEN_GAP)
		this.addNeuronRow(rMlp, hidden, sx, Y.hidden, HIDDEN_GAP)
		this.addConnections(rMlp, outFlows, sx, Y.hidden, HIDDEN_GAP, Y.block1, NEURON_GAP)
		this.addLetterRows(rMlp, vm.blocks[0].output, Y.block1)
		this.addLabel(rMlp, Y.hidden + 0.6, "④ MLP", `${d} → ${4 * d} → ${d} neurons per character`, "#3d8bfd")

		// ⑤ later blocks, compact: attention arcs + every letter's output
		if (vm.blocks.length > 1) {
			if (vm.first > 0) this.addEarlierMarker(rBlock2, Y.block1)
			this.addArcs(rBlock2, vm.blocks[1].arcs, Y.block1, Y.block2)
			this.addLetterRows(rBlock2, vm.blocks[vm.blocks.length - 1].output, Y.block2)
			const more = vm.blocks.length > 2 ? ` (to ${vm.blocks.length})` : ""
			this.addLabel(rBlock2, Y.block2, `⑤ Block 2${more}`, "attention + MLP again, on block 1's result", "#2ecc71")
		}

		// ⑥ output: the most likely characters after the selected letter
		const outLines: number[] = []
		vm.top.forEach((t, k) => {
			const x = sx + (k - (vm.top.length - 1) / 2) * 0.75
			const sphere = new THREE.Mesh(
				new THREE.SphereGeometry(0.1 + 0.3 * Math.sqrt(t.p), 20, 14),
				new THREE.MeshLambertMaterial({ color: k === 0 ? "#ff9f43" : "#3a5a8c", transparent: true }),
			)
			sphere.position.set(x, Y.output, 0)
			rOut.add(sphere)
			const label = textSprite(`${t.char} ${Math.round(t.p * 100)}%`, k === 0 ? "#ffb347" : "#c8d0de", 0.32)
			label.position.set(x, Y.output + 0.62, 0)
			rOut.add(label)
			outLines.push(sx, top + 0.1, 0, x, Y.output - 0.15, 0)
		})
		this.addSegments(rOut, outLines, "#ff9f43", 0.5)
		this.addLabel(rOut, Y.output, "⑥ Output", "one neuron per character → probabilities", "#ff9f43")
		this.addLabel(rPick, Y.pick, "⑦ Pick", "draw one character at random", "#ffb347")

		this.applyStep()
	}

	/** Show the picked character above the output (step ⑦), or remove it. */
	showPick(char: string | null): void {
		const row = this.rows[6]
		const vm = this.vm
		if (!row || !vm) return
		for (const child of [...row.children]) {
			if (child.userData.pick) {
				disposeObject(child)
				row.remove(child)
			}
		}
		if (char === null) return
		const sprite = textSprite(`picked "${char}"`, "#ffb347", 0.55)
		sprite.position.set(letterX(vm.selected, vm.letters.length), Y.pick, 0)
		sprite.userData.pick = true
		row.add(sprite)
		this.applyStep()
	}

	/**
	 * While `stepping`, light up the row of `step` (finished rows dimmed, upcoming faint) and optionally fly
	 * the camera there; otherwise every row is fully lit.
	 */
	setStep(step: number, stepping: boolean, flyMs: number | null): void {
		this.step = step
		this.stepping = stepping
		this.applyStep()
		if (stepping && flyMs !== null && this.vm) {
			// From the front, slightly left and above, looking between the middle and the selected letter.
			// +3 shifts the network left on screen, so the row labels clear the caption panel on the right.
			const x = letterX(this.vm.selected, this.vm.letters.length) * 0.45 + 3
			const y = STEP_Y[step]
			this.host.flyTo({ x: x - 2.5, y: y + 2, z: 19 }, { x, y: y + 0.5, z: 0 }, flyMs)
		}
	}

	/** Camera that shows the whole network. */
	cameraPosition(): THREE.Vector3 {
		return new THREE.Vector3(1, 8, 27)
	}

	cameraTarget(): THREE.Vector3 {
		return new THREE.Vector3(5, 5.5, 0)
	}

	pickables(): THREE.Object3D[] {
		return this.pickable
	}

	pick(hit: THREE.Intersection | null): void {
		if (!hit) return
		const ud = hit.object.userData
		let letter: number | null = null
		if (typeof ud.letter === "number") letter = ud.letter
		else if (typeof ud.perLetter === "number" && hit.instanceId !== undefined) {
			letter = Math.floor(hit.instanceId / ud.perLetter)
		}
		if (letter !== null) this.onSelect?.(letter)
	}

	private applyStep(): void {
		this.rows.forEach((row, i) => {
			const level = rowBrightness(i, this.step, this.stepping)
			row.traverse((obj) => {
				const material = (obj as THREE.Mesh).material as THREE.Material | THREE.Material[] | undefined
				if (!material || Array.isArray(material)) return
				material.userData.base ??= material.opacity
				material.transparent = true
				material.opacity = material.userData.base * level
			})
		})
	}

	/** Neuron rows of all visible letters (one instanced mesh), colored by value; clicking selects a letter. */
	private addLetterRows(group: THREE.Group, values: number[][], y: number): void {
		const n = values.length
		const per = values[0]?.length ?? 0
		if (!per) return
		const mesh = new THREE.InstancedMesh(
			new THREE.SphereGeometry(NEURON_R, 10, 8),
			new THREE.MeshLambertMaterial(),
			n * per,
		)
		let max = 1e-6
		for (const row of values) for (const v of row) max = Math.max(max, Math.abs(v))
		values.forEach((row, i) => {
			const cx = letterX(i, n)
			row.forEach((v, k) => {
				this.matrix.makeTranslation(cx + (k - (per - 1) / 2) * NEURON_GAP, y, 0)
				mesh.setMatrixAt(i * per + k, this.matrix)
				mesh.setColorAt(i * per + k, heat(0.5 + (0.5 * v) / max, this.color))
			})
		})
		mesh.userData.perLetter = per
		group.add(mesh)
		this.pickable.push(mesh)
	}

	/** One neuron row centred at x. */
	private addNeuronRow(group: THREE.Group, values: number[], x: number, y: number, gap: number): void {
		if (!values.length) return
		const mesh = new THREE.InstancedMesh(
			new THREE.SphereGeometry(NEURON_R * 1.15, 10, 8),
			new THREE.MeshLambertMaterial(),
			values.length,
		)
		const max = Math.max(1e-6, ...values.map(Math.abs))
		values.forEach((v, k) => {
			this.matrix.makeTranslation(x + (k - (values.length - 1) / 2) * gap, y, 0)
			mesh.setMatrixAt(k, this.matrix)
			mesh.setColorAt(k, heat(0.5 + (0.5 * v) / max, this.color))
		})
		group.add(mesh)
	}

	/**
	 * Connections from one neuron row to the next, colored by what flows along them (value × weight),
	 * like the digit view: orange pushes the next neuron up, blue pushes it down, dark ≈ nothing.
	 * flows[a][b] goes from neuron a of the row at fromY to neuron b of the row at toY.
	 */
	private addConnections(
		group: THREE.Group,
		flows: number[][],
		x: number,
		fromY: number,
		fromGap: number,
		toY: number,
		toGap: number,
	): void {
		const positions: number[] = []
		const colors: number[] = []
		let max = 1e-9
		for (const row of flows) for (const f of row) max = Math.max(max, Math.abs(f))
		flows.forEach((row, a) => {
			const ax = x + (a - (flows.length - 1) / 2) * fromGap
			row.forEach((f, b) => {
				const bx = x + (b - (row.length - 1) / 2) * toGap
				positions.push(ax, fromY, 0, bx, toY, 0)
				const t = Math.abs(f) / max
				this.color.set(f >= 0 ? POSITIVE : NEGATIVE).multiplyScalar(0.03 + 0.97 * t * t)
				colors.push(this.color.r, this.color.g, this.color.b, this.color.r, this.color.g, this.color.b)
			})
		})
		const geometry = new THREE.BufferGeometry()
		geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3))
		geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3))
		const material = new THREE.LineBasicMaterial({
			vertexColors: true,
			blending: THREE.AdditiveBlending,
			depthWrite: false,
		})
		group.add(new THREE.LineSegments(geometry, material))
	}

	private addSegments(group: THREE.Group, positions: number[], color: string, opacity: number): void {
		const geometry = new THREE.BufferGeometry()
		geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3))
		group.add(new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color, transparent: true, opacity })))
	}

	/** Attention arcs from the source row into the selected letter's row; % on the two strongest. */
	private addArcs(group: THREE.Group, arcs: ViewArc[], fromY: number, toY: number): void {
		const vm = this.vm
		if (!vm) return
		const n = vm.letters.length
		const to = { x: letterX(vm.selected, n), y: toY - 0.1, z: 0 }
		arcs.forEach((arc, k) => {
			const fromX = arc.from === "earlier" ? letterX(-1, n) : letterX(arc.from, n)
			const points = arcPoints({ x: fromX, y: fromY + 0.1, z: 0 }, to, 0.6 + 0.35 * arc.head)
			const color = HEAD_COLORS[arc.head % HEAD_COLORS.length]
			const geometry = new LineGeometry()
			geometry.setPositions(points)
			const material = new LineMaterial({
				color,
				linewidth: 1.5 + 7 * arc.weight,
				transparent: true,
				opacity: 0.35 + 0.65 * arc.weight,
				depthWrite: false,
			})
			material.resolution.copy(this.host.size)
			group.add(new Line2(geometry, material))
			if (k < 2) {
				const mid = Math.floor(points.length / 6) * 3 // middle point of the curve
				const label = textSprite(`${Math.round(arc.weight * 100)}%`, color, 0.34)
				label.position.set(points[mid], points[mid + 1] + 0.25, points[mid + 2])
				group.add(label)
			}
		})
	}

	private addEarlierMarker(group: THREE.Group, y: number): void {
		const sprite = textSprite("… earlier text", "#8b95a7", 0.3, false)
		sprite.position.set(letterX(-1, this.vm?.letters.length ?? VISIBLE_LETTERS), y, 0)
		group.add(sprite)
	}

	private addLabel(group: THREE.Group, y: number, title: string, sub: string, color: string): void {
		const label = labelSprite(title, sub, color)
		label.position.set(letterX(VISIBLE_LETTERS - 1, VISIBLE_LETTERS) + 4.8, y, 0)
		group.add(label)
	}

	private clear(): void {
		for (const child of [...this.root.children]) {
			child.traverse(disposeObject)
			this.root.remove(child)
		}
		this.rows = []
		this.pickable = []
	}
}
