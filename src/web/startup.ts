export interface StartupResult<T> {
	loaded: T | null
	failures: { entry: T; error: Error }[]
}

/**
 * Try entries in order until one loads, so a single broken model (e.g. a corrupt
 * model.json that is also the remembered choice) can't lock the whole viewer.
 * Stops early when `shouldContinue` says so (e.g. the user picked a model meanwhile).
 */
export async function loadFirstWorking<T>(
	entries: readonly T[],
	load: (entry: T) => Promise<void>,
	shouldContinue: () => boolean = () => true,
): Promise<StartupResult<T>> {
	const failures: StartupResult<T>["failures"] = []
	for (const entry of entries) {
		if (!shouldContinue()) break
		try {
			await load(entry)
			return { loaded: entry, failures }
		} catch (err) {
			failures.push({ entry, error: err instanceof Error ? err : new Error(String(err)) })
		}
	}
	return { loaded: null, failures }
}
