/** Images-seen points for early snapshots: dense at the start, where the network changes fastest. */
export const EARLY_MILESTONES = [0, 50, 100, 250, 500, 1_000, 2_500, 5_000, 10_000, 20_000, 40_000]

/** Early milestones strictly inside the first epoch (each epoch end gets its own snapshot). */
export function earlyMilestones(epochSize: number): number[] {
	return EARLY_MILESTONES.filter((m) => m < epochSize)
}

/** Tells the training loop when a snapshot is due; several points crossed in one batch → one snapshot. */
export class SnapshotSchedule {
	private next = 0
	private readonly points: readonly number[]

	constructor(points: readonly number[]) {
		this.points = points
	}

	due(imagesSeen: number): boolean {
		let hit = false
		while (this.next < this.points.length && this.points[this.next] <= imagesSeen) {
			this.next++
			hit = true
		}
		return hit
	}
}
