import { decodeFloat16, encodeFloat16 } from "./float16"
import { isObject, type SnapshotFile } from "./model-format"

/** Snapshot v2: every learnable tensor in Network.params() order. */
export function encodeSnapshot(params: readonly Float32Array[]): SnapshotFile {
	return { params: params.map((p) => encodeFloat16(p)) }
}

/**
 * Decode a snapshot — v2 `{ params }` or v1 `{ layers: [{ W, b }] }` — and check
 * every tensor against the sizes the network expects.
 */
export function decodeSnapshot(file: unknown, expectedSizes: readonly number[]): Float32Array[] {
	if (!isObject(file)) throw new Error("Invalid snapshot: not an object")
	let encoded: unknown[]
	if (Array.isArray(file.params)) encoded = file.params
	else if (Array.isArray(file.layers))
		encoded = (file.layers as unknown[]).flatMap((l) => (isObject(l) ? [l.W, l.b] : [l, l]))
	else throw new Error("Invalid snapshot: missing params")
	if (encoded.length !== expectedSizes.length) {
		throw new Error(`Invalid snapshot: expected ${expectedSizes.length} tensors, got ${encoded.length}`)
	}
	return encoded.map((text, i) => {
		if (typeof text !== "string") throw new Error(`Invalid snapshot: tensor ${i} is malformed`)
		const values = decodeFloat16(text)
		if (values.length !== expectedSizes[i]) {
			throw new Error(`Invalid snapshot: tensor ${i} has ${values.length} values, expected ${expectedSizes[i]}`)
		}
		return values
	})
}
