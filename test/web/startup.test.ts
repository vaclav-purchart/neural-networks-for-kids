import { describe, expect, it } from "vitest"
import { loadFirstWorking } from "../../src/web/startup"

describe("loadFirstWorking", () => {
	it("falls back to the next model when the preferred one fails to load", async () => {
		const tried: string[] = []
		const result = await loadFirstWorking(["broken", "good", "other"], async (name) => {
			tried.push(name)
			if (name === "broken") throw new Error("Invalid model.json: not an object")
		})
		expect(result.loaded).toBe("good")
		expect(tried).toEqual(["broken", "good"])
		expect(result.failures.map((f) => [f.entry, f.error.message])).toEqual([
			["broken", "Invalid model.json: not an object"],
		])
	})

	it("reports every failure when nothing loads", async () => {
		const result = await loadFirstWorking(["a", "b"], async () => {
			throw new Error("HTTP 404")
		})
		expect(result.loaded).toBeNull()
		expect(result.failures).toHaveLength(2)
	})
})

describe("loadFirstWorking when the user takes over", () => {
	it("stops trying fallbacks once the user has picked a model themselves", async () => {
		let userPicked = false
		const tried: string[] = []
		const result = await loadFirstWorking(
			["broken", "good"],
			async (name) => {
				tried.push(name)
				userPicked = true // the user switches models while this one is loading…
				throw new Error("corrupt") // …and then it fails
			},
			() => !userPicked,
		)
		expect(tried).toEqual(["broken"])
		expect(result.loaded).toBeNull()
	})
})
