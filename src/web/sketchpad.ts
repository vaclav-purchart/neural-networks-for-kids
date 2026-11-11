type Mode = "draw" | "erase"

/** 28×28 drawing grid with a soft round brush. Values are 0..1, row-major. */
export class Sketchpad {
	readonly rows: number
	readonly cols: number
	brushRadius = 1.4
	onChange: (() => void) | null = null
	private readonly values: Float32Array
	private readonly canvas: HTMLCanvasElement
	private readonly ctx: CanvasRenderingContext2D
	private mode: Mode | null = null
	private last: { x: number; y: number } | null = null

	constructor(host: HTMLElement, rows = 28, cols = 28) {
		this.rows = rows
		this.cols = cols
		this.values = new Float32Array(rows * cols)
		this.canvas = document.createElement("canvas")
		this.canvas.width = cols * 10
		this.canvas.height = rows * 10
		const ctx = this.canvas.getContext("2d")
		if (!ctx) throw new Error("Canvas 2D is not available")
		this.ctx = ctx
		host.append(this.canvas)

		this.canvas.addEventListener("contextmenu", (e) => e.preventDefault())
		this.canvas.addEventListener("pointerdown", (e) => {
			e.preventDefault()
			this.canvas.setPointerCapture(e.pointerId)
			this.mode = e.button === 2 ? "erase" : "draw"
			const p = this.toCell(e)
			this.last = p
			this.stamp(p)
			this.changed()
		})
		this.canvas.addEventListener("pointermove", (e) => {
			if (!this.mode) return
			this.strokeTo(this.toCell(e))
			this.changed()
		})
		const stop = () => {
			this.mode = null
			this.last = null
		}
		this.canvas.addEventListener("pointerup", stop)
		this.canvas.addEventListener("pointercancel", stop)
		this.render()
	}

	getPixels(): Float32Array {
		return this.values.slice()
	}

	setPixels(pixels: ArrayLike<number>): void {
		this.values.set(pixels)
		this.changed()
	}

	clear(): void {
		this.values.fill(0)
		this.changed()
	}

	private toCell(e: PointerEvent): { x: number; y: number } {
		const r = this.canvas.getBoundingClientRect()
		return { x: ((e.clientX - r.left) / r.width) * this.cols, y: ((e.clientY - r.top) / r.height) * this.rows }
	}

	/** Stamp along the segment from the last point so fast strokes have no gaps. */
	private strokeTo(p: { x: number; y: number }): void {
		const from = this.last ?? p
		const steps = Math.max(1, Math.ceil(Math.hypot(p.x - from.x, p.y - from.y) / 0.35))
		for (let s = 1; s <= steps; s++) {
			const t = s / steps
			this.stamp({ x: from.x + (p.x - from.x) * t, y: from.y + (p.y - from.y) * t })
		}
		this.last = p
	}

	private stamp(p: { x: number; y: number }): void {
		const erase = this.mode === "erase"
		const r = erase ? this.brushRadius * 1.6 : this.brushRadius
		const y0 = Math.max(0, Math.floor(p.y - r))
		const y1 = Math.min(this.rows - 1, Math.ceil(p.y + r))
		const x0 = Math.max(0, Math.floor(p.x - r))
		const x1 = Math.min(this.cols - 1, Math.ceil(p.x + r))
		for (let cy = y0; cy <= y1; cy++) {
			for (let cx = x0; cx <= x1; cx++) {
				const d = Math.hypot(cx + 0.5 - p.x, cy + 0.5 - p.y)
				if (d > r) continue
				const falloff = 1 - (d / r) ** 2 // soft edge
				const i = cy * this.cols + cx
				if (erase) this.values[i] *= 1 - falloff
				else this.values[i] = Math.max(this.values[i], Math.min(1, falloff * 1.3))
			}
		}
	}

	private changed(): void {
		this.render()
		this.onChange?.()
	}

	private render(): void {
		const { ctx, rows, cols } = this
		const cw = this.canvas.width / cols
		const ch = this.canvas.height / rows
		ctx.fillStyle = "#05070c"
		ctx.fillRect(0, 0, this.canvas.width, this.canvas.height)
		for (let r = 0; r < rows; r++) {
			for (let c = 0; c < cols; c++) {
				const v = this.values[r * cols + c]
				if (v <= 0.01) continue
				const g = Math.round(v * 255)
				ctx.fillStyle = `rgb(${g},${g},${g})`
				ctx.fillRect(c * cw, r * ch, cw, ch)
			}
		}
		ctx.strokeStyle = "rgba(255,255,255,0.05)"
		ctx.lineWidth = 1
		ctx.beginPath()
		for (let i = 1; i < cols; i++) {
			ctx.moveTo(i * cw + 0.5, 0)
			ctx.lineTo(i * cw + 0.5, this.canvas.height)
		}
		for (let i = 1; i < rows; i++) {
			ctx.moveTo(0, i * ch + 0.5)
			ctx.lineTo(this.canvas.width, i * ch + 0.5)
		}
		ctx.stroke()
	}
}
