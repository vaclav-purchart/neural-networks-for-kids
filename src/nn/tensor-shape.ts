/** Channels × height × width. Dense vectors are { c: n, h: 1, w: 1 }. */
export interface Shape {
	c: number
	h: number
	w: number
}

export const shapeSize = (s: Shape): number => s.c * s.h * s.w

export const vectorShape = (n: number): Shape => ({ c: n, h: 1, w: 1 })

/** "64" for vectors, "8×26×26" for images / feature maps. */
export const describeShape = (s: Shape): string => (s.h === 1 && s.w === 1 ? String(s.c) : `${s.c}×${s.h}×${s.w}`)
