import { describe, expect, it } from "vitest"
import { describeImageSnapshot, describeTextSnapshot } from "../../src/web/timeline"

describe("timeline labels", () => {
	it("describes image snapshots as before", () => {
		const s = { file: "", imagesSeen: 60000, epoch: 1, testAccuracy: 0.9512, testSamples: 10000, avgLoss: 0.2 }
		expect(describeImageSnapshot(s, 2, 10)).toBe(
			"Snapshot 3/10 · 60,000 images seen (1.00 epochs) · test accuracy 95.1% on 10,000 images · loss 0.200",
		)
	})
	it("describes text snapshots with characters, loss and accuracy", () => {
		const s = {
			file: "",
			steps: 500,
			charsSeen: 1024000,
			testLoss: 1.6234,
			testAccuracy: 0.512,
			testSamples: 16384,
			avgLoss: 1.7,
			sample: "",
		}
		expect(describeTextSnapshot(s, 4, 20)).toBe(
			"Snapshot 5/20 · step 500 · 1,024,000 characters seen · test loss 1.623 · next-character accuracy 51.2%",
		)
	})
})
