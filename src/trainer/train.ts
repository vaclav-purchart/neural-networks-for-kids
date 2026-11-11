import { performance } from "node:perf_hooks"
import { type Dataset, loadDataset, type Split } from "../data/load"
import { Adam } from "../nn/adam"
import { buildNetwork, cnnSpecs, mlpSpecs } from "../nn/architectures"
import type { Network } from "../nn/network"
import { mulberry32, shuffleInPlace } from "../nn/rng"
import { describeShape } from "../nn/tensor-shape"
import { type ImageTrainOptions, parseTrainArgs, type TrainOptions, USAGE } from "./cli"
import { pickSamples, type TrainedSnapshot, writeModel } from "./export"
import { earlyMilestones, SnapshotSchedule } from "./timeline"
import { trainText } from "./train-text"

/** Early snapshots measure accuracy on this many test images to keep training fast. */
const QUICK_EVAL_SAMPLES = 2000

/** Copy `count` images (chosen by `pick`) into x as normalised floats, labels into y. */
function loadBatch(
	ds: Dataset,
	split: Split,
	pick: (k: number) => number,
	count: number,
	x: Float32Array,
	y: Uint8Array,
): void {
	const size = ds.rows * ds.cols
	const scale = 1 / (255 * ds.std)
	const shift = ds.mean / ds.std // (v/255 - mean) / std
	for (let k = 0; k < count; k++) {
		const i = pick(k)
		y[k] = split.labels[i]
		const src = i * size
		const dst = k * size
		for (let p = 0; p < size; p++) x[dst + p] = split.images[src + p] * scale - shift
	}
}

function evaluate(net: Network, ds: Dataset, split: Split, maxSamples: number): { accuracy: number; samples: number } {
	const n = Math.min(split.count, maxSamples)
	const size = ds.rows * ds.cols
	const classes = ds.spec.labels.length
	const B = 500
	const x = new Float32Array(B * size)
	const y = new Uint8Array(B)
	let correct = 0
	for (let start = 0; start < n; start += B) {
		const count = Math.min(B, n - start)
		loadBatch(ds, split, (k) => start + k, count, x, y)
		const probs = net.forward(x.subarray(0, count * size), count).act.at(-1)!
		for (let k = 0; k < count; k++) {
			let best = 0
			for (let c = 1; c < classes; c++) if (probs[k * classes + c] > probs[k * classes + best]) best = c
			if (best === y[k]) correct++
		}
	}
	return { accuracy: n ? correct / n : 0, samples: n }
}

/** e.g. "1×28×28 → conv 8×26×26 → pool 8×13×13 → … → 10" */
function describeNetwork(net: Network): string {
	const parts = net.layers.map((l) =>
		l.kind === "dense"
			? describeShape(l.outShape)
			: `${l.kind === "conv2d" ? "conv" : "pool"} ${describeShape(l.outShape)}`,
	)
	return [describeShape(net.inputShape), ...parts].join(" → ")
}

async function main(): Promise<void> {
	let opts: TrainOptions
	try {
		opts = parseTrainArgs(process.argv.slice(2))
	} catch (err) {
		console.error(`${(err as Error).message}\n\n${USAGE}`)
		process.exit(1)
	}
	if (opts.kind === "text") await trainText(opts)
	else await trainImage(opts)
}

async function trainImage(opts: ImageTrainOptions): Promise<void> {
	const ds = await loadDataset(opts.dataset, opts)
	const size = ds.rows * ds.cols
	const classes = ds.spec.labels.length
	const layers = opts.arch === "cnn" ? cnnSpecs(opts.filters, opts.hidden, classes) : mlpSpecs(opts.hidden, classes)
	let net: Network
	try {
		net = buildNetwork({ c: 1, h: ds.rows, w: ds.cols }, layers)
	} catch (err) {
		console.error(`These layer sizes don't fit ${ds.rows}×${ds.cols} images: ${(err as Error).message}\n\n${USAGE}`)
		process.exit(1)
	}
	console.log(
		`${ds.spec.title}: ${ds.train.count} train / ${ds.test.count} test images, network (${opts.arch.toUpperCase()}) ${describeNetwork(net)}`,
	)

	const rand = mulberry32(opts.seed)
	net.init(rand)
	const adam = new Adam(net.params(), { lr: opts.lr })
	const snapshots: TrainedSnapshot[] = []
	let imagesSeen = 0
	let lossSum = 0
	let lossCount = 0

	const takeSnapshot = (full: boolean): number => {
		const { accuracy, samples } = evaluate(net, ds, ds.test, full ? Infinity : QUICK_EVAL_SAMPLES)
		snapshots.push({
			meta: {
				imagesSeen,
				epoch: imagesSeen / ds.train.count,
				testAccuracy: accuracy,
				testSamples: samples,
				avgLoss: lossCount ? lossSum / lossCount : null,
			},
			params: net.params().map((p) => p.slice()),
		})
		lossSum = 0
		lossCount = 0
		return accuracy
	}

	const schedule = new SnapshotSchedule(earlyMilestones(ds.train.count))
	if (schedule.due(0)) takeSnapshot(false)

	const order = Uint32Array.from({ length: ds.train.count }, (_, i) => i)
	const x = new Float32Array(opts.batch * size)
	const y = new Uint8Array(opts.batch)
	for (let epoch = 1; epoch <= opts.epochs; epoch++) {
		const started = performance.now()
		shuffleInPlace(order, rand)
		let epochLoss = 0
		for (let start = 0; start < ds.train.count; start += opts.batch) {
			const count = Math.min(opts.batch, ds.train.count - start)
			loadBatch(ds, ds.train, (k) => order[start + k], count, x, y)
			const labels = y.subarray(0, count)
			const result = net.forward(x.subarray(0, count * size), count)
			const loss = net.loss(result, labels)
			adam.step(net.backward(result, labels))
			imagesSeen += count
			lossSum += loss * count
			lossCount += count
			epochLoss += loss * count
			if (epoch === 1 && schedule.due(imagesSeen)) takeSnapshot(false)
		}
		const accuracy = takeSnapshot(true)
		const seconds = ((performance.now() - started) / 1000).toFixed(1)
		console.log(
			`epoch ${epoch}/${opts.epochs}  loss ${(epochLoss / ds.train.count).toFixed(4)}  test accuracy ${(accuracy * 100).toFixed(2)}%  ${seconds}s`,
		)
	}

	const samples = pickSamples(ds.test, classes, size, opts.samplesPerClass, rand)
	const info = await writeModel({ outDir: opts.outDir, dataset: ds, arch: opts.arch, layers, snapshots, samples })
	console.log(`\nSaved ${info.snapshots.length} snapshots to ${opts.outDir}/${info.id}/`)
	console.log("Start the viewer with: npm run dev")
}

main().catch((err: unknown) => {
	console.error(err instanceof Error ? err.message : err)
	process.exit(1)
})
