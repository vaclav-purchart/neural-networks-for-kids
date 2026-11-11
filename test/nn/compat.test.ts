import { readFileSync } from "node:fs"
import { describe, expect, it } from "vitest"
import { buildNetwork, mlpSpecs } from "../../src/nn/architectures"
import { decodeSnapshot } from "../../src/nn/serialize"
import { vectorShape } from "../../src/nn/tensor-shape"

const fixture = JSON.parse(readFileSync(new URL("../fixtures/v1-mlp.json", import.meta.url), "utf8")) as {
	sizes: number[]
	snapshot: unknown
	input: number[]
	act: number[][]
}

describe("v1 MLP compatibility", () => {
	it("gives exactly the outputs the old Mlp code produced", () => {
		const { sizes } = fixture
		const net = buildNetwork(vectorShape(sizes[0]), mlpSpecs(sizes.slice(1, -1), sizes[sizes.length - 1]))
		net.setParams(
			decodeSnapshot(
				fixture.snapshot,
				net.params().map((p) => p.length),
			),
		)
		const act = net.forward(Float32Array.from(fixture.input), 1).act.slice(1)
		act.forEach((a, i) => {
			a.forEach((v, j) => {
				expect(v).toBeCloseTo(fixture.act[i][j], 6)
			})
		})
	})
})
