import { describe, expect, it } from "vitest"
import {
	buildVocab,
	decode,
	displayChar,
	encode,
	normalizeText,
	splitIds,
	vocabFromChars,
} from "../../../src/nn/transformer/tokenizer"

describe("normalizeText", () => {
	it("strips a BOM and turns CRLF / CR into LF", () => {
		expect(normalizeText("﻿a\r\nb\rc\n")).toBe("a\nb\nc\n")
	})
})

describe("buildVocab", () => {
	it("lists the distinct characters sorted, including diacritics", () => {
		expect(buildVocab("čaba\n").chars).toEqual(["\n", "a", "b", "č"])
	})
	it("rejects text with fewer than 2 distinct characters", () => {
		expect(() => buildVocab("aaaa")).toThrow(/at least 2 distinct/)
	})
	it("rejects more than 1000 distinct characters", () => {
		const text = Array.from({ length: 1001 }, (_, i) => String.fromCodePoint(0x4e00 + i)).join("")
		expect(() => buildVocab(text)).toThrow(/1000/)
	})
})

describe("encode / decode", () => {
	const vocab = vocabFromChars(["\n", " ", "a", "b"])
	it("round-trips known text", () => {
		const { ids, ignored } = encode("ab a\n", vocab)
		expect(ids).toEqual([2, 3, 1, 2, 0])
		expect(ignored).toEqual([])
		expect(decode(ids, vocab)).toBe("ab a\n")
	})
	it("drops unknown characters and lists each once, in order", () => {
		const { ids, ignored } = encode("aXbYX", vocab)
		expect(ids).toEqual([2, 3])
		expect(ignored).toEqual(["X", "Y"])
	})
	it("treats astral characters as one character", () => {
		expect(encode("a😀", vocab).ignored).toEqual(["😀"])
	})
})

describe("displayChar", () => {
	it("makes whitespace visible", () => {
		expect([" ", "\n", "\t", "x"].map(displayChar)).toEqual(["␣", "↵", "⇥", "x"])
	})
})

describe("splitIds", () => {
	it("puts the first 90% in train and the rest in test", () => {
		const ids = Uint16Array.from({ length: 100 }, (_, i) => i % 3)
		const { train, test } = splitIds(ids, 4)
		expect(train.length).toBe(90)
		expect(test.length).toBe(10)
		expect(test[0]).toBe(ids[90])
	})
	it("rejects text whose test split can't hold one context window", () => {
		const ids = new Uint16Array(60)
		expect(() => splitIds(ids, 8)).toThrow(/too short/)
	})
})
