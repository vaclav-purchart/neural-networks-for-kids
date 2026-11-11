import { describe, expect, it } from "vitest"
import { DEFAULT_TEXT_STEPS, parseTrainArgs } from "../../src/trainer/cli"

describe("parseTrainArgs", () => {
	it("applies per-architecture defaults", () => {
		const mlp = parseTrainArgs(["--dataset", "mnist"])
		if (mlp.kind !== "image") throw new Error("expected image options")
		expect(mlp).toMatchObject({
			kind: "image",
			dataset: "mnist",
			arch: "mlp",
			hidden: [128, 64],
			filters: [],
			epochs: 5,
			batch: 64,
			lr: 0.001,
			seed: 42,
			offline: false,
		})
		expect(mlp.limit).toBeUndefined()
		expect(parseTrainArgs(["--dataset", "mnist", "--arch", "cnn"])).toMatchObject({
			arch: "cnn",
			filters: [8, 16],
			hidden: [64],
			epochs: 3,
		})
	})

	it("parses --filters for CNNs", () => {
		expect(
			parseTrainArgs(["--dataset", "mnist", "--arch", "cnn", "--filters", "32, 64", "--hidden", "128"]),
		).toMatchObject({
			filters: [32, 64],
			hidden: [128],
		})
	})

	it("rejects bad architecture options", () => {
		expect(() => parseTrainArgs(["--dataset", "mnist", "--arch", "rnn"])).toThrow(
			/--arch must be mlp or cnn, got "rnn"/,
		)
		expect(() => parseTrainArgs(["--dataset", "mnist", "--filters", "8"])).toThrow(
			/--filters only applies to --arch cnn/,
		)
		expect(() => parseTrainArgs(["--dataset", "mnist", "--arch", "cnn", "--filters", "8,0"])).toThrow(
			/--filters must be a positive integer, got "0"/,
		)
	})

	it("parses all options", () => {
		const o = parseTrainArgs([
			"--dataset",
			"emnist-letters",
			"--hidden",
			"256, 128",
			"--epochs",
			"2",
			"--limit",
			"500",
			"--offline",
		])
		expect(o).toMatchObject({ dataset: "emnist-letters", hidden: [256, 128], epochs: 2, limit: 500, offline: true })
	})

	it("rejects bad input with a helpful message", () => {
		expect(() => parseTrainArgs([])).toThrow(/--dataset must be one of: mnist, emnist-letters/)
		expect(() => parseTrainArgs(["--dataset", "cifar"])).toThrow(/--dataset must be one of/)
		expect(() => parseTrainArgs(["--dataset", "mnist", "--hidden", "128,abc"])).toThrow(
			/--hidden must be a positive integer, got "abc"/,
		)
		expect(() => parseTrainArgs(["--dataset", "mnist", "--bogus"])).toThrow()
	})
})

describe("parseTrainArgs (text)", () => {
	it("uses the transformer defaults for a text dataset", () => {
		expect(parseTrainArgs(["--dataset", "shakespeare"])).toEqual({
			kind: "text",
			dataset: "shakespeare",
			file: undefined,
			context: 64,
			dModel: 64,
			heads: 4,
			blocks: 2,
			steps: DEFAULT_TEXT_STEPS,
			batch: 32,
			lr: 0.003,
			seed: 42,
			offline: false,
			dataDir: "data",
			outDir: "public/models",
		})
	})

	it("parses overrides and --file", () => {
		const o = parseTrainArgs([
			"--dataset",
			"text",
			"--file",
			"book.txt",
			"--context",
			"32",
			"--heads",
			"2",
			"--steps",
			"100",
		])
		expect(o).toMatchObject({ kind: "text", file: "book.txt", context: 32, heads: 2, steps: 100 })
	})

	it("requires --file for the text dataset and rejects it elsewhere", () => {
		expect(() => parseTrainArgs(["--dataset", "text"])).toThrow(/--file/)
		expect(() => parseTrainArgs(["--dataset", "names", "--file", "x.txt"])).toThrow(/--file/)
		expect(() => parseTrainArgs(["--dataset", "mnist", "--file", "x.txt"])).toThrow(/--file/)
	})

	it("rejects image-only flags for text and text-only flags for images", () => {
		expect(() => parseTrainArgs(["--dataset", "names", "--hidden", "64"])).toThrow(/--hidden/)
		expect(() => parseTrainArgs(["--dataset", "names", "--epochs", "2"])).toThrow(/--epochs/)
		expect(() => parseTrainArgs(["--dataset", "mnist", "--steps", "10"])).toThrow(/--steps/)
		expect(() => parseTrainArgs(["--dataset", "mnist", "--arch", "transformer"])).toThrow(/transformer/)
		expect(() => parseTrainArgs(["--dataset", "names", "--arch", "cnn"])).toThrow(/transformer/)
	})

	it("rejects a width the heads don't divide", () => {
		expect(() => parseTrainArgs(["--dataset", "names", "--d-model", "30", "--heads", "4"])).toThrow(/divisible/)
	})
})
