import * as THREE from "three"
import { Dense } from "../nn/layers"
import type { Network } from "../nn/network"
import type { NeuronRef } from "./neuron-panel"
import { disposeObject, heat, makeTextSprite, type SceneContent } from "./scene-host"

const LAYER_GAP = 7
const INPUT_SPACING = 0.28
const NODE_SPACING = 0.75
const OUTPUT_SPACING = 0.9
/** Output layers with many classes (e.g. 26 letters) become a grid; spread it out so labels fit. */
const OUTPUT_GRID_SPACING = 1.4
/** Side length of one feature-map square, and the gap between squares. */
const MAP_SIZE = 2
const MAP_GAP = 0.35

const POSITIVE = new THREE.Color(0xff7a45)
const NEGATIVE = new THREE.Color(0x3fa7ff)

/** n items in a centred grid in the y/z plane at x; rows run downwards, columns along +z. */
function gridLayout(n: number, x: number, cols: number, spacing: number): Float32Array {
	const rows = Math.ceil(n / cols)
	const pos = new Float32Array(n * 3)
	for (let i = 0; i < n; i++) {
		const r = Math.floor(i / cols)
		const c = i % cols
		pos[3 * i] = x
		pos[3 * i + 1] = ((rows - 1) / 2 - r) * spacing
		pos[3 * i + 2] = (c - (cols - 1) / 2) * spacing
	}
	return pos
}

/** A square outline in the y/z plane (x = 0), side `size`. */
function squareOutline(size: number, color: number): THREE.LineLoop {
	const h = size / 2
	const geometry = new THREE.BufferGeometry().setFromPoints([
		new THREE.Vector3(0, -h, -h),
		new THREE.Vector3(0, h, -h),
		new THREE.Vector3(0, h, h),
		new THREE.Vector3(0, -h, h),
	])
	return new THREE.LineLoop(geometry, new THREE.LineBasicMaterial({ color }))
}

/** Indices of the k largest |values|, largest first. */
function topIndicesByAbs(values: Float32Array, k: number): number[] {
	const idx = Array.from({ length: values.length }, (_, i) => i)
	idx.sort((a, b) => Math.abs(values[b]) - Math.abs(values[a]))
	return idx.slice(0, k)
}

interface FeatureMapView {
	mesh: THREE.Mesh
	texture: THREE.DataTexture
	data: Uint8Array
}

/** One column of the 3D view. Max-pool layers are not drawn. */
interface DisplayColumn {
	/** Index into ForwardResult.act (0 = input). */
	act: number
	kind: "input" | "dense" | "conv"
	/** Centre of every node (input / dense) or feature map (conv), 3 floats each. */
	positions: Float32Array
	nodes: THREE.InstancedMesh | null
	maps: FeatureMapView[]
	/** Feature-map size (conv columns). */
	h: number
	w: number
}

interface Connections {
	lines: THREE.LineSegments
	/** act index of the source column. */
	sourceAct: number
	src: Uint32Array
	weight: Float32Array
}

export class NetworkScene implements SceneContent {
	onNeuronClick: ((ref: NeuronRef | null) => void) | null = null
	readonly root = new THREE.Group()
	private readonly marker: THREE.Mesh
	private readonly mapMarker: THREE.LineLoop
	private readonly color = new THREE.Color()
	private readonly matrix = new THREE.Matrix4()
	private columns: DisplayColumn[] = []
	private connections: Connections[] = []
	private view = new THREE.Vector3(-10, 5, 15)

	constructor() {
		this.marker = new THREE.Mesh(
			new THREE.SphereGeometry(1, 16, 12),
			new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true }),
		)
		this.mapMarker = squareOutline(MAP_SIZE * 1.18, 0xffffff)
		this.marker.visible = false
		this.mapMarker.visible = false
		this.root.add(this.marker, this.mapMarker)
	}

	/** Where the camera should start for the current model. */
	cameraPosition(): THREE.Vector3 {
		return this.view.clone()
	}

	setModel(net: Network, labels: readonly string[], inputShape: readonly [number, number]): void {
		this.clear()
		const specs: { act: number; kind: DisplayColumn["kind"]; count: number; h: number; w: number }[] = [
			{ act: 0, kind: "input", count: inputShape[0] * inputShape[1], h: inputShape[0], w: inputShape[1] },
		]
		net.layers.forEach((layer, i) => {
			if (layer.kind === "maxpool") return
			const { c, h, w } = layer.outShape
			specs.push({ act: i + 1, kind: layer.kind === "conv2d" ? "conv" : "dense", count: c, h, w })
		})
		const D = specs.length
		this.columns = specs.map((spec, d): DisplayColumn => {
			const x = (d - (D - 1) / 2) * LAYER_GAP
			const base = { act: spec.act, kind: spec.kind, h: spec.h, w: spec.w }
			if (spec.kind === "input") {
				const positions = gridLayout(spec.count, x, spec.w, INPUT_SPACING)
				const geometry = new THREE.BoxGeometry(0.05, INPUT_SPACING * 0.85, INPUT_SPACING * 0.85)
				return { ...base, positions, nodes: this.makeNodes(positions, geometry, spec.act), maps: [] }
			}
			if (spec.kind === "conv") {
				const positions = gridLayout(spec.count, x, Math.ceil(Math.sqrt(spec.count)), MAP_SIZE + MAP_GAP)
				const maps = Array.from({ length: spec.count }, (_, c) => this.makeMap(positions, c, spec.h, spec.w, spec.act))
				return { ...base, positions, nodes: null, maps }
			}
			const isOutput = d === D - 1
			const cols = isOutput && spec.count <= 12 ? 1 : Math.ceil(Math.sqrt(spec.count))
			const spacing = !isOutput ? NODE_SPACING : cols === 1 ? OUTPUT_SPACING : OUTPUT_GRID_SPACING
			const positions = gridLayout(spec.count, x, cols, spacing)
			const geometry = new THREE.SphereGeometry(isOutput ? 0.3 : 0.2, 14, 10)
			return { ...base, positions, nodes: this.makeNodes(positions, geometry, spec.act), maps: [] }
		})

		const out = this.columns[D - 1].positions
		const outputIsColumn = labels.length <= 12
		labels.forEach((label, i) => {
			const sprite = makeTextSprite(label)
			// Beside a single column; above each node when the outputs form a grid.
			const [dx, dy] = outputIsColumn ? [0.8, 0] : [0, 0.6]
			sprite.position.set(out[3 * i] + dx, out[3 * i + 1] + dy, out[3 * i + 2])
			this.root.add(sprite)
		})

		const width = (D - 1) * LAYER_GAP
		this.view = new THREE.Vector3(-width * 0.55 - 4, width * 0.2 + 3, width * 0.75 + 6)
	}

	/** Lines for the strongest incoming weights of every dense neuron whose input column is drawn. */
	setWeights(net: Network, opts: { perNeuron: number; minWeight: number }): void {
		for (const conn of this.connections) {
			this.root.remove(conn.lines)
			disposeObject(conn.lines)
		}
		this.connections = []
		this.columns.forEach((target, d) => {
			if (d === 0 || target.kind !== "dense") return
			const source = this.columns[d - 1]
			// Only when the previous drawn column is this layer's direct input and has node positions
			// (a CNN's first dense layer reads pooled maps, which aren't drawn).
			if (source.act !== target.act - 1 || source.kind === "conv") return
			const layer = net.layers[target.act - 1]
			if (!(layer instanceof Dense)) return
			const src: number[] = []
			const dst: number[] = []
			const weight: number[] = []
			for (let j = 0; j < layer.outDim; j++) {
				const row = layer.W.subarray(j * layer.inDim, (j + 1) * layer.inDim)
				for (const i of topIndicesByAbs(row, opts.perNeuron)) {
					if (Math.abs(row[i]) < opts.minWeight) break // sorted, so the rest are smaller
					src.push(i)
					dst.push(j)
					weight.push(row[i])
				}
			}
			const a = source.positions
			const b = target.positions
			const positions = new Float32Array(src.length * 6)
			for (let k = 0; k < src.length; k++) {
				positions.set([a[3 * src[k]], a[3 * src[k] + 1], a[3 * src[k] + 2]], 6 * k)
				positions.set([b[3 * dst[k]], b[3 * dst[k] + 1], b[3 * dst[k] + 2]], 6 * k + 3)
			}
			const geometry = new THREE.BufferGeometry()
			geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3))
			geometry.setAttribute("color", new THREE.BufferAttribute(new Float32Array(src.length * 6), 3))
			const material = new THREE.LineBasicMaterial({
				vertexColors: true,
				transparent: true,
				// The input layer has by far the most lines; keep it from saturating to white.
				opacity: source.kind === "input" ? 0.35 : 0.8,
				blending: THREE.AdditiveBlending,
				depthWrite: false,
			})
			const lines = new THREE.LineSegments(geometry, material)
			this.root.add(lines)
			this.connections.push({
				lines,
				sourceAct: source.act,
				src: Uint32Array.from(src),
				weight: Float32Array.from(weight),
			})
		})
	}

	/**
	 * Colour nodes and feature maps by activation, and connections by what actually flows along them
	 * (source activation × weight; warm = excites, cool = inhibits, dark ≈ nothing).
	 */
	update(act: readonly Float32Array[], inputPixels: ArrayLike<number>): void {
		const last = this.columns.length - 1
		this.columns.forEach((col, d) => {
			const values = col.kind === "input" ? inputPixels : act[col.act]
			if (col.kind === "conv") {
				this.paintMaps(col, values)
				return
			}
			let max = 1
			if (col.kind === "dense" && d !== last) {
				max = 1e-6
				for (let i = 0; i < values.length; i++) max = Math.max(max, values[i])
			}
			const mesh = col.nodes!
			for (let i = 0; i < values.length; i++) mesh.setColorAt(i, heat(values[i] / max, this.color))
			if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true
		})
		for (const conn of this.connections) {
			const source = act[conn.sourceAct]
			const n = conn.src.length
			const flow = new Float32Array(n)
			let max = 1e-6
			for (let k = 0; k < n; k++) {
				flow[k] = source[conn.src[k]] * conn.weight[k]
				max = Math.max(max, Math.abs(flow[k]))
			}
			const attr = conn.lines.geometry.getAttribute("color") as THREE.BufferAttribute
			const colors = attr.array as Float32Array
			for (let k = 0; k < n; k++) {
				const base = flow[k] >= 0 ? POSITIVE : NEGATIVE
				const t = Math.abs(flow[k]) / max
				const s = 0.02 + 0.98 * t * t // squared: strong flows stand out, weak ones fade
				for (let v = 0; v < 2; v++) {
					colors[6 * k + 3 * v] = base.r * s
					colors[6 * k + 3 * v + 1] = base.g * s
					colors[6 * k + 3 * v + 2] = base.b * s
				}
			}
			attr.needsUpdate = true
		}
	}

	highlight(ref: NeuronRef | null): void {
		this.marker.visible = false
		this.mapMarker.visible = false
		const col = ref ? this.columns.find((c) => c.act === ref.layer) : undefined
		if (!ref || !col) return
		const p = col.positions
		const i = ref.index
		if (col.kind === "conv") {
			this.mapMarker.position.set(p[3 * i], p[3 * i + 1], p[3 * i + 2])
			this.mapMarker.visible = true
		} else {
			this.marker.position.set(p[3 * i], p[3 * i + 1], p[3 * i + 2])
			this.marker.scale.setScalar(col.kind === "input" ? 0.22 : 0.42)
			this.marker.visible = true
		}
	}

	private makeNodes(positions: Float32Array, geometry: THREE.BufferGeometry, act: number): THREE.InstancedMesh {
		const n = positions.length / 3
		const mesh = new THREE.InstancedMesh(geometry, new THREE.MeshLambertMaterial(), n)
		for (let i = 0; i < n; i++) {
			this.matrix.makeTranslation(positions[3 * i], positions[3 * i + 1], positions[3 * i + 2])
			mesh.setMatrixAt(i, this.matrix)
			mesh.setColorAt(i, heat(0, this.color))
		}
		mesh.userData.layer = act
		this.root.add(mesh)
		return mesh
	}

	private makeMap(positions: Float32Array, c: number, h: number, w: number, act: number): FeatureMapView {
		const data = new Uint8Array(h * w * 4)
		const texture = new THREE.DataTexture(data, w, h, THREE.RGBAFormat)
		texture.magFilter = THREE.NearestFilter
		texture.minFilter = THREE.NearestFilter
		texture.needsUpdate = true
		const mesh = new THREE.Mesh(
			new THREE.PlaneGeometry(MAP_SIZE, MAP_SIZE),
			new THREE.MeshBasicMaterial({ map: texture, side: THREE.DoubleSide }),
		)
		// Lie in the y/z plane like the input layer: texture columns along +z, rows downwards.
		mesh.rotation.y = -Math.PI / 2
		mesh.position.set(positions[3 * c], positions[3 * c + 1], positions[3 * c + 2])
		mesh.userData.layer = act
		mesh.userData.index = c
		const outline = squareOutline(MAP_SIZE * 1.04, 0x2d3850)
		outline.rotation.y = Math.PI / 2 // undo the plane's rotation: the outline is defined in the y/z plane
		mesh.add(outline)
		this.root.add(mesh)
		return { mesh, texture, data }
	}

	/** Write one layer's activations into its textures, scaled to the layer's maximum. */
	private paintMaps(col: DisplayColumn, values: ArrayLike<number>): void {
		const { h, w } = col
		const plane = h * w
		let max = 1e-6
		for (let i = 0; i < col.maps.length * plane; i++) max = Math.max(max, values[i])
		col.maps.forEach((map, c) => {
			for (let r = 0; r < h; r++) {
				for (let x = 0; x < w; x++) {
					heat(values[c * plane + r * w + x] / max, this.color)
					const o = ((h - 1 - r) * w + x) * 4 // texture row 0 is the bottom of the square
					map.data[o] = Math.round(this.color.r * 255)
					map.data[o + 1] = Math.round(this.color.g * 255)
					map.data[o + 2] = Math.round(this.color.b * 255)
					map.data[o + 3] = 255
				}
			}
			map.texture.needsUpdate = true
		})
	}

	pickables(): THREE.Object3D[] {
		return this.columns.flatMap((c): THREE.Object3D[] => (c.nodes ? [c.nodes] : c.maps.map((m) => m.mesh)))
	}

	pick(hit: THREE.Intersection | null): void {
		let ref: NeuronRef | null = null
		if (hit) {
			const index = hit.object instanceof THREE.InstancedMesh ? hit.instanceId : (hit.object.userData.index as number)
			if (index !== undefined) ref = { layer: hit.object.userData.layer as number, index }
		}
		this.highlight(ref)
		this.onNeuronClick?.(ref)
	}

	private clear(): void {
		for (const child of [...this.root.children]) {
			if (child === this.marker || child === this.mapMarker) continue
			child.traverse(disposeObject)
			this.root.remove(child)
		}
		this.columns = []
		this.connections = []
		this.marker.visible = false
		this.mapMarker.visible = false
	}
}
