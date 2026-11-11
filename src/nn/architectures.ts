import { Conv2d, Dense, type Layer, MaxPool } from "./layers"
import { Network } from "./network"
import type { Shape } from "./tensor-shape"

export type Arch = "mlp" | "cnn"

/** Layer descriptions as stored in model.json. */
export type LayerSpec =
	| { type: "dense"; units: number; activation: "relu" | "softmax" }
	| { type: "conv2d"; filters: number; kernel: 3; activation: "relu" }
	| { type: "maxpool"; size: 2 }

/** Dense hidden layers (ReLU) followed by a softmax output. */
export function mlpSpecs(hidden: readonly number[], classes: number): LayerSpec[] {
	return [
		...hidden.map((units): LayerSpec => ({ type: "dense", units, activation: "relu" })),
		{ type: "dense", units: classes, activation: "softmax" },
	]
}

/** One conv(3×3)+ReLU → max-pool block per entry of `filters`, then the MLP head. */
export function cnnSpecs(filters: readonly number[], hidden: readonly number[], classes: number): LayerSpec[] {
	return [
		...filters.flatMap((f): LayerSpec[] => [
			{ type: "conv2d", filters: f, kernel: 3, activation: "relu" },
			{ type: "maxpool", size: 2 },
		]),
		...mlpSpecs(hidden, classes),
	]
}

/** Build layers from specs; any shape problem is reported as "layer i (type): reason". */
export function buildNetwork(inputShape: Shape, specs: readonly LayerSpec[]): Network {
	const layers: Layer[] = []
	let shape = inputShape
	specs.forEach((spec, i) => {
		try {
			if (spec.type !== "dense" && layers.some((l) => l.kind === "dense")) {
				throw new Error("conv2d/maxpool layers must come before dense layers")
			}
			const layer: Layer =
				spec.type === "dense"
					? new Dense(shape, spec.units, spec.activation)
					: spec.type === "conv2d"
						? new Conv2d(shape, spec.filters, spec.activation)
						: new MaxPool(shape)
			layers.push(layer)
			shape = layer.outShape
		} catch (err) {
			throw new Error(`layer ${i} (${spec.type}): ${(err as Error).message}`)
		}
	})
	return new Network(inputShape, layers)
}
