import * as THREE from "three"
import { OrbitControls } from "three/addons/controls/OrbitControls.js"
import { wheelAction, zoomFactor } from "./wheel"

const COLD = new THREE.Color(0x1d2a44)
const MID = new THREE.Color(0x3d8bfd)
const HOT = new THREE.Color(0xffb347)

/** 0 → dark blue, 0.5 → blue, 1 → orange. */
export function heat(t: number, out: THREE.Color): THREE.Color {
	const v = Math.min(1, Math.max(0, t))
	return v < 0.5 ? out.copy(COLD).lerp(MID, v * 2) : out.copy(MID).lerp(HOT, (v - 0.5) * 2)
}

/** A text label that always faces the camera. `width` is the canvas width in multiples of 128 px. */
export function makeTextSprite(
	text: string,
	opts: { width?: number; scale?: number; color?: string } = {},
): THREE.Sprite {
	const width = opts.width ?? 1
	const canvas = document.createElement("canvas")
	canvas.width = 128 * width
	canvas.height = 128
	const ctx = canvas.getContext("2d")!
	ctx.fillStyle = opts.color ?? "#e8eefc"
	ctx.font = "bold 88px system-ui, sans-serif"
	ctx.textAlign = "center"
	ctx.textBaseline = "middle"
	ctx.fillText(text, canvas.width / 2, 70)
	const texture = new THREE.CanvasTexture(canvas)
	texture.colorSpace = THREE.SRGBColorSpace
	const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, depthWrite: false }))
	const s = opts.scale ?? 0.8
	sprite.scale.set(s * width, s, s)
	return sprite
}

export function disposeObject(obj: THREE.Object3D): void {
	if (obj instanceof THREE.Mesh || obj instanceof THREE.Line || obj instanceof THREE.Sprite) {
		obj.geometry.dispose()
		const material = obj.material as THREE.Material & { map?: THREE.Texture | null }
		material.map?.dispose()
		material.dispose()
	}
	if (obj instanceof THREE.InstancedMesh) obj.dispose()
}

/** What the host shows: a group of objects plus click handling. */
export interface SceneContent {
	readonly root: THREE.Group
	/** Objects a click can hit. */
	pickables(): THREE.Object3D[]
	/** A click that wasn't a drag: the nearest hit, or null for empty space. */
	pick(hit: THREE.Intersection | null): void
}

/** Renderer, camera, orbit controls, render loop and click picking shared by the image and text views. */
export class SceneHost {
	readonly camera = new THREE.PerspectiveCamera(50, 1, 0.1, 500)
	readonly controls: OrbitControls
	/** Canvas size in CSS pixels (fat lines need it). */
	readonly size = new THREE.Vector2(1, 1)
	private readonly canvas: HTMLCanvasElement
	private readonly renderer: THREE.WebGLRenderer
	private readonly scene = new THREE.Scene()
	private readonly raycaster = new THREE.Raycaster()
	private content: SceneContent | null = null
	private flight: {
		fromPos: THREE.Vector3
		fromTarget: THREE.Vector3
		toPos: THREE.Vector3
		toTarget: THREE.Vector3
		start: number
		ms: number
	} | null = null

	constructor(canvas: HTMLCanvasElement) {
		this.canvas = canvas
		this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true })
		this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
		this.scene.background = new THREE.Color(0x0b0f17)
		this.scene.add(new THREE.AmbientLight(0xffffff, 0.7))
		const light = new THREE.DirectionalLight(0xffffff, 1.4)
		light.position.set(5, 10, 7)
		this.scene.add(light)
		this.controls = new OrbitControls(this.camera, canvas)
		this.controls.enableDamping = true
		this.setupPicking()
		// Before OrbitControls' own wheel zoom (capture phase): touchpad swipes pan, pinch / mouse wheel zoom.
		canvas.addEventListener("wheel", (e) => this.onWheel(e), { capture: true, passive: false })
		new ResizeObserver(() => this.resize()).observe(canvas)
		this.resize()
		this.renderer.setAnimationLoop((now) => {
			this.stepFlight(now)
			this.controls.update()
			this.renderer.render(this.scene, this.camera)
		})
	}

	/** Show one content; the previous one is taken out of the scene, not disposed. */
	show(content: SceneContent): void {
		if (this.content === content) return
		if (this.content) this.scene.remove(this.content.root)
		this.content = content
		this.scene.add(content.root)
	}

	frame(position: THREE.Vector3Like, target: THREE.Vector3Like = { x: 0, y: 0, z: 0 }): void {
		this.flight = null
		this.camera.position.set(position.x, position.y, position.z)
		this.controls.target.set(target.x, target.y, target.z)
		this.controls.update()
	}

	/** Move the camera smoothly to look at `target` from `position` over `ms` milliseconds. */
	flyTo(position: THREE.Vector3Like, target: THREE.Vector3Like, ms: number): void {
		if (ms <= 0) {
			this.frame(position, target)
			return
		}
		this.flight = {
			fromPos: this.camera.position.clone(),
			fromTarget: this.controls.target.clone(),
			toPos: new THREE.Vector3(position.x, position.y, position.z),
			toTarget: new THREE.Vector3(target.x, target.y, target.z),
			start: performance.now(),
			ms,
		}
	}

	private stepFlight(now: number): void {
		const f = this.flight
		if (!f) return
		const t = Math.min(1, (now - f.start) / f.ms)
		const e = t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2 // ease in-out
		this.camera.position.lerpVectors(f.fromPos, f.toPos, e)
		this.controls.target.lerpVectors(f.fromTarget, f.toTarget, e)
		if (t >= 1) this.flight = null
	}

	private onWheel(e: WheelEvent): void {
		e.preventDefault()
		e.stopImmediatePropagation()
		this.flight = null // the user takes over the camera
		const camera = this.camera
		const target = this.controls.target
		if (wheelAction(e) === "zoom") {
			const offset = camera.position.clone().sub(target)
			const distance = offset.length()
			const next = Math.min(200, Math.max(1, distance * zoomFactor(e.deltaY, e.ctrlKey)))
			camera.position.copy(target).add(offset.multiplyScalar(next / distance))
		} else {
			// Move camera and target together, so the scene follows the fingers like a scrolled page.
			const distance = camera.position.distanceTo(target)
			const perPixel = (2 * distance * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2)) / this.size.y
			const right = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 0).multiplyScalar(e.deltaX * perPixel)
			const up = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 1).multiplyScalar(-e.deltaY * perPixel)
			const move = right.add(up)
			camera.position.add(move)
			target.add(move)
		}
		this.controls.update()
	}

	private setupPicking(): void {
		let down: { x: number; y: number } | null = null
		this.canvas.addEventListener("pointerdown", (e) => {
			down = e.button === 0 ? { x: e.clientX, y: e.clientY } : null
		})
		this.canvas.addEventListener("pointerup", (e) => {
			const start = down
			down = null
			if (!start || !this.content || Math.hypot(e.clientX - start.x, e.clientY - start.y) > 5) return // a drag
			const rect = this.canvas.getBoundingClientRect()
			const ndc = new THREE.Vector2(
				((e.clientX - rect.left) / rect.width) * 2 - 1,
				-((e.clientY - rect.top) / rect.height) * 2 + 1,
			)
			this.raycaster.setFromCamera(ndc, this.camera)
			this.content.pick(this.raycaster.intersectObjects(this.content.pickables(), false)[0] ?? null)
		})
	}

	private resize(): void {
		const w = this.canvas.clientWidth
		const h = this.canvas.clientHeight
		if (!w || !h) return
		this.size.set(w, h)
		this.renderer.setSize(w, h, false)
		this.camera.aspect = w / h
		this.camera.updateProjectionMatrix()
	}
}
