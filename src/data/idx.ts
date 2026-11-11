/** IDX: the simple binary format MNIST and EMNIST are distributed in. */
export interface IdxArray {
	shape: number[]
	data: Uint8Array
}

/** Parse an IDX file of unsigned bytes. `name` is used in error messages. */
export function parseIdx(buf: Uint8Array, name = "IDX file"): IdxArray {
	if (buf.length < 4) throw new Error(`${name}: too short for an IDX header`)
	if (buf[0] !== 0 || buf[1] !== 0) throw new Error(`${name}: bad magic number`)
	if (buf[2] !== 0x08) {
		throw new Error(`${name}: unsupported data type 0x${buf[2].toString(16)} (expected unsigned byte 0x08)`)
	}
	const ndim = buf[3]
	if (ndim < 1) throw new Error(`${name}: no dimensions`)
	const headerLength = 4 + 4 * ndim
	if (buf.length < headerLength) throw new Error(`${name}: truncated header`)
	const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength)
	const shape: number[] = []
	for (let d = 0; d < ndim; d++) shape.push(view.getUint32(4 + 4 * d, false))
	const expected = shape.reduce((a, b) => a * b, 1)
	const data = buf.subarray(headerLength)
	if (data.length !== expected) {
		throw new Error(`${name}: expected ${expected} bytes of data for shape [${shape.join(", ")}], found ${data.length}`)
	}
	return { shape, data }
}
