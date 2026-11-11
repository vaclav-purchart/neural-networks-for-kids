import { performance } from "node:perf_hooks"
import { loadTextCorpus } from "../data/text"
import { Adam } from "../nn/adam"
import type { TextSnapshotMeta } from "../nn/model-format"
import { mulberry32 } from "../nn/rng"
import { Transformer, type TransformerConfig, transformerParamSizes } from "../nn/transformer/model"
import { generate, startId } from "../nn/transformer/sample"
import { buildVocab, decode, encode, splitIds } from "../nn/transformer/tokenizer"
import type { TextTrainOptions } from "./cli"
import { writeTextModel } from "./export"
import { evaluateText, evalWindows, fillBatch, lrAt, promptIds, snapshotSteps } from "./text-training"
import { SnapshotSchedule } from "./timeline"

const QUICK_EVAL_WINDOWS = 64
const FULL_EVAL_WINDOWS = 256
const SAMPLE_LENGTH = 200
const SAMPLE_TEMPERATURE = 0.8
const SAMPLE_SEED = 1234
const LOG_EVERY = 50

export async function trainText(opts: TextTrainOptions): Promise<void> {
	const corpus = await loadTextCorpus(opts)
	const vocab = buildVocab(corpus.text)
	const ids = Uint16Array.from(encode(corpus.text, vocab).ids)
	const T = opts.context
	const { train, test } = splitIds(ids, T)
	const config: TransformerConfig = {
		vocab: vocab.chars.length,
		context: T,
		dModel: opts.dModel,
		heads: opts.heads,
		blocks: opts.blocks,
	}
	const model = new Transformer(config)
	const weights = transformerParamSizes(config).reduce((a, b) => a + b)
	const n = (v: number) => v.toLocaleString("en-US")
	console.log(
		`${corpus.title}: ${n(train.length)} train / ${n(test.length)} test characters, ${vocab.chars.length} distinct; ` +
			`transformer ${opts.blocks} blocks × ${opts.heads} heads, d ${opts.dModel}, context ${T}, ${n(weights)} weights`,
	)

	const rand = mulberry32(opts.seed)
	model.init(rand)
	const adam = new Adam(model.params(), { lr: opts.lr })
	const quick = evalWindows(test.length, T, QUICK_EVAL_WINDOWS)
	const full = evalWindows(test.length, T, FULL_EVAL_WINDOWS)
	const prompt = promptIds(test, vocab)
	const fallback = startId(vocab)
	const snapshots: { meta: Omit<TextSnapshotMeta, "file">; params: Float32Array[] }[] = []
	let lossSum = 0
	let lossCount = 0

	const takeSnapshot = (step: number): number => {
		const ev = evaluateText(model, test, step === opts.steps ? full : quick, T)
		const generated = generate(model, prompt, SAMPLE_LENGTH, SAMPLE_TEMPERATURE, mulberry32(SAMPLE_SEED), fallback)
		snapshots.push({
			meta: {
				steps: step,
				charsSeen: step * opts.batch * T,
				testLoss: ev.loss,
				testAccuracy: ev.accuracy,
				testSamples: ev.samples,
				avgLoss: lossCount ? lossSum / lossCount : null,
				sample: decode([...prompt, ...generated], vocab),
			},
			params: model.params().map((p) => p.slice()),
		})
		lossSum = 0
		lossCount = 0
		return ev.loss
	}

	const schedule = new SnapshotSchedule(snapshotSteps(opts.steps))
	if (schedule.due(0)) takeSnapshot(0)
	const x = new Int32Array(opts.batch * T)
	const y = new Int32Array(opts.batch * T)
	const starts = new Int32Array(opts.batch)
	const started = performance.now()
	let recent = 0
	let recentCount = 0
	for (let step = 1; step <= opts.steps; step++) {
		for (let k = 0; k < opts.batch; k++) starts[k] = Math.floor(rand() * (train.length - T))
		fillBatch(train, starts, T, x, y)
		const fwd = model.forward(x, opts.batch, T)
		const loss = model.loss(fwd, y)
		adam.lr = lrAt(step, opts.steps, opts.lr)
		adam.step(model.backward(fwd, y))
		lossSum += loss
		lossCount++
		recent += loss
		recentCount++
		if (step % LOG_EVERY === 0 || step === opts.steps) {
			const elapsed = (performance.now() - started) / 1000
			const eta = (elapsed / step) * (opts.steps - step)
			console.log(
				`step ${step}/${opts.steps}  loss ${(recent / recentCount).toFixed(3)}  ` +
					`${(elapsed / step).toFixed(2)}s/step  ~${Math.ceil(eta / 60)} min left`,
			)
			recent = 0
			recentCount = 0
		}
		if (schedule.due(step)) {
			const testLoss = takeSnapshot(step)
			console.log(`  snapshot ${snapshots.length}: test loss ${testLoss.toFixed(3)}`)
		}
	}
	const last = snapshots[snapshots.length - 1].meta
	console.log(
		`\nFinal test loss ${last.testLoss.toFixed(3)}, next-character accuracy ${(last.testAccuracy * 100).toFixed(1)}%`,
	)
	console.log(`Sample:\n${last.sample}\n`)
	await writeTextModel({
		outDir: opts.outDir,
		id: corpus.id,
		dataset: opts.dataset,
		title: corpus.title,
		vocab: vocab.chars,
		config,
		snapshots,
	})
	console.log(`Saved ${snapshots.length} snapshots to ${opts.outDir}/${corpus.id}/`)
	console.log("Start the viewer with: npm run dev")
}
