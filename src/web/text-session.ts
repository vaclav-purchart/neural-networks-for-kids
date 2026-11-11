import { type TextModelInfo, textConfig } from "../nn/model-format"
import type { Rand } from "../nn/rng"
import { Transformer, type TransformerForward } from "../nn/transformer/model"
import { sampleFromLogits, startId } from "../nn/transformer/sample"
import { encode, type Vocab, vocabFromChars } from "../nn/transformer/tokenizer"

export interface SessionRead {
	/** Token ids fed to the model: the last `context` known characters (or the start character). */
	ids: number[]
	/** Characters of the text the model doesn't know, once each. */
	ignored: string[]
	fwd: TransformerForward
}

/** One text model in the viewer: its weights, the last forward pass and sampling. No DOM. */
export class TextSession {
	readonly net: Transformer
	readonly vocab: Vocab
	private weightsLoaded = false
	private last: SessionRead | null = null

	constructor(info: TextModelInfo) {
		this.net = new Transformer(textConfig(info))
		this.vocab = vocabFromChars(info.labels)
	}

	setWeights(params: Float32Array[]): void {
		this.net.setParams(params)
		this.weightsLoaded = true
	}

	/** Run the model on the end of `text`; null until weights are loaded. */
	read(text: string): SessionRead | null {
		if (!this.weightsLoaded) return null
		const { ids, ignored } = encode(text, this.vocab)
		let window = ids.slice(-this.net.config.context)
		if (window.length === 0) window = [startId(this.vocab)]
		this.last = { ids: window, ignored, fwd: this.net.forward(window, 1, window.length) }
		return this.last
	}

	/** What to put into a prompt with no known characters so generation starts at a line start, else "". */
	seed(text: string): string {
		return encode(text, this.vocab).ids.length ? "" : this.vocab.chars[startId(this.vocab)]
	}

	/** `text` with a generated character added; text without known characters first gets the start newline. */
	continueText(text: string, char: string): string {
		return text + this.seed(text) + char
	}

	/** A next character for the text of the last read(), or null if nothing has been read yet. */
	next(temperature: number, rand: Rand): string | null {
		const last = this.last
		return last ? (this.pickAt(last.ids.length - 1, temperature, rand)?.char ?? null) : null
	}

	/** Sample what follows `position` (an index into the last read's ids), with its probability. */
	pickAt(position: number, temperature: number, rand: Rand): { id: number; char: string; p: number } | null {
		const last = this.last
		if (!last || position < 0 || position >= last.ids.length) return null
		const V = this.net.config.vocab
		const id = sampleFromLogits(last.fwd.logits, position * V, V, temperature, rand)
		return { id, char: this.vocab.chars[id], p: last.fwd.probs[position * V + id] }
	}
}
