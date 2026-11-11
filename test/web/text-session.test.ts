import { describe, expect, it } from "vitest"
import type { TextModelInfo } from "../../src/nn/model-format"
import { mulberry32 } from "../../src/nn/rng"
import { Transformer } from "../../src/nn/transformer/model"
import { TextSession } from "../../src/web/text-session"

function info(labels: string[]): TextModelInfo {
	return {
		formatVersion: 2,
		id: "t",
		arch: "transformer",
		dataset: "names",
		title: "T",
		labels,
		transformer: { context: 4, dModel: 8, heads: 2, blocks: 1 },
		trainedAt: "2026-10-15T00:00:00.000Z",
		snapshots: [],
	}
}

function weights(vocab: number): Float32Array[] {
	const net = new Transformer({ vocab, context: 4, dModel: 8, heads: 2, blocks: 1 })
	net.init(mulberry32(1))
	return net.params()
}

describe("TextSession", () => {
	it("doesn't generate before its own weights are loaded (no state from a previous model)", () => {
		const first = new TextSession(info(["\n", "a", "b"]))
		first.setWeights(weights(3))
		expect(first.read("ab")).not.toBeNull()
		expect(first.next(0.8, mulberry32(1))).not.toBeNull()

		const second = new TextSession(info(["\n", "a", "b", "c", "d"]))
		expect(second.read("ab")).toBeNull()
		expect(second.next(0.8, mulberry32(1))).toBeNull()
	})

	it("seeds an empty prompt with a newline so generation starts at a line start", () => {
		const session = new TextSession(info(["\n", "a", "b"]))
		expect(session.seed("")).toBe("\n")
		expect(session.seed("XYZ")).toBe("\n") // only unknown characters: same as empty
		expect(session.seed("ab")).toBe("")
	})

	it("feeds only the last `context` known characters and lists the unknown ones", () => {
		const session = new TextSession(info(["\n", "a", "b"]))
		session.setWeights(weights(3))
		const read = session.read("ab\nabXba")!
		expect(read.ids).toEqual([1, 2, 2, 1]) // "abba": the last 4 of a, b, ↵, a, b, b, a
		expect(read.ignored).toEqual(["X"])
		expect(read.fwd.T).toBe(4)
	})

	it("picks a next character for any position of the last read, with its probability", () => {
		const session = new TextSession(info(["\n", "a", "b"]))
		expect(session.pickAt(0, 0.8, mulberry32(1))).toBeNull() // nothing read yet
		session.setWeights(weights(3))
		const read = session.read("abab")!
		const greedy = session.pickAt(1, 0, mulberry32(1))!
		const probs = read.fwd.probs.subarray(3, 6)
		expect(greedy.p).toBeCloseTo(Math.max(...probs))
		expect(greedy.char).toBe(["\n", "a", "b"][greedy.id])
		expect(session.pickAt(9, 0.8, mulberry32(1))).toBeNull() // outside the read text
	})

	it("continues an empty text after the start newline, so it stays in the model's context", () => {
		const session = new TextSession(info(["\n", "a", "b"]))
		expect(session.continueText("", "a")).toBe("\na")
		expect(session.continueText("XY", "a")).toBe("XY\na") // only unknown characters: same as empty
		expect(session.continueText("ab", "a")).toBe("aba")
	})
})
