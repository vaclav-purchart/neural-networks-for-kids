import type { LoadedModel } from "./model-store"

/** What main.ts needs from the image and text views. */
export interface ViewMode {
	/** Show this mode's panels and build its scene for a freshly loaded model (throws if it can't). */
	enter(model: LoadedModel): void
	/** Weights of the selected snapshot, in parameter order. */
	setWeights(params: Float32Array[]): void
	/** Hide this mode's panels and stop anything running. */
	leave(): void
}
