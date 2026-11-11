/** Character-level tokens: every distinct character of the training text is one token. */

export const MAX_VOCAB = 1000

export interface Vocab {
	chars: string[]
	index: Map<string, number>
}

/** Strip a byte-order mark and use \n line endings everywhere. */
export function normalizeText(raw: string): string {
	return raw.replace(/^﻿/, "").replace(/\r\n?/g, "\n")
}

export function vocabFromChars(chars: string[]): Vocab {
	return { chars, index: new Map(chars.map((c, i) => [c, i])) }
}

/** Sorted distinct characters (code points, so "č" or "😀" is one character). */
export function buildVocab(text: string): Vocab {
	const chars = [...new Set(Array.from(text))].sort()
	if (chars.length < 2) throw new Error("The text needs at least 2 distinct characters")
	if (chars.length > MAX_VOCAB) {
		throw new Error(`The text has ${chars.length} distinct characters; at most ${MAX_VOCAB} are supported`)
	}
	return vocabFromChars(chars)
}

/** Token ids of the known characters; unknown ones are skipped and listed once each, in order. */
export function encode(text: string, vocab: Vocab): { ids: number[]; ignored: string[] } {
	const ids: number[] = []
	const ignored: string[] = []
	for (const c of text) {
		const id = vocab.index.get(c)
		if (id !== undefined) ids.push(id)
		else if (!ignored.includes(c)) ignored.push(c)
	}
	return { ids, ignored }
}

export function decode(ids: ArrayLike<number>, vocab: Vocab): string {
	let out = ""
	for (let i = 0; i < ids.length; i++) out += vocab.chars[ids[i]] ?? ""
	return out
}

/** Visible stand-ins for whitespace in labels and charts. */
export function displayChar(c: string): string {
	if (c === " ") return "␣"
	if (c === "\n") return "↵"
	if (c === "\t") return "⇥"
	return c
}

/** First 90% train, last 10% test; both must hold at least one window of context + 1 characters. */
export function splitIds(ids: Uint16Array, context: number): { train: Uint16Array; test: Uint16Array } {
	const cut = Math.floor(ids.length * 0.9)
	const train = ids.subarray(0, cut)
	const test = ids.subarray(cut)
	if (train.length < context + 1 || test.length < context + 1) {
		const needed = Math.ceil((context + 1) / 0.1)
		throw new Error(`The text is too short: ${ids.length} characters, need at least ${needed} for context ${context}`)
	}
	return { train, test }
}
