import { describe, expect, it } from "vitest"
import { EARLY_MILESTONES, earlyMilestones, SnapshotSchedule } from "../../src/trainer/timeline"

describe("earlyMilestones", () => {
	it("keeps only points inside the first epoch", () => {
		expect(earlyMilestones(60_000)).toEqual(EARLY_MILESTONES)
		expect(earlyMilestones(2_500)).toEqual([0, 50, 100, 250, 500, 1_000])
	})
})

describe("SnapshotSchedule", () => {
	it("fires once per crossing, even when a batch jumps over several points", () => {
		const s = new SnapshotSchedule([0, 50, 100, 250, 500])
		expect(s.due(0)).toBe(true)
		expect(s.due(64)).toBe(true)
		expect(s.due(64)).toBe(false)
		expect(s.due(90)).toBe(false)
		expect(s.due(600)).toBe(true) // 100, 250, 500 all passed → one snapshot
		expect(s.due(10_000)).toBe(false)
	})
})
