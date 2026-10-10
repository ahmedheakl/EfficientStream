// The streaming session (efficientstream/session.py): feed frames as they arrive, ask questions at any moment.
//
// Per stored frame (one every 1 / grid_fps seconds): resize to store size, JPEG round trip, one MobileCLIP2-S0
// image embedding and, on the phone, the frame's thumbnail through the MiniCPM vision tower (the reference
// computes the same thumbnail features at question time; doing it once here moves that work off the question).
// At question time: one text embedding, a top-k lookup, the latest frame through the vision tower, one prefill.

import Foundation
import MLX

public struct StreamResult: Sendable {
    /// Option letter or open-ended answer.
    public var text: String
    public var probs: [Double]?
    /// Times of the thumbnails shown (the latest frame is `now`).
    public var frameTimes: [Double]
    public var now: Double
    public var promptTokens: Int
    /// Question -> answer.
    public var seconds: Double
    /// Question -> first answer text (multiple choice: the answer).
    public var secondsToFirstToken: Double
    public var timings: QuestionTimings
}

/// Where the time of one question goes, in seconds (in order). `total` is question -> answer.
public struct QuestionTimings: Codable, Sendable {
    /// MobileCLIP2-S0 text embedding of the question.
    public var textEmbed = 0.0
    /// Lazy mode only: MobileCLIP2-S0 image embeddings of the stored frames, at question time.
    public var frameEmbeds = 0.0
    /// Similarity + top-k.
    public var topK = 0.0
    /// Thumbnails through the vision tower (0 when computed at store time).
    public var thumbnails = 0.0
    /// The latest frame: resize to 896 + JPEG round trip.
    public var latestPrep = 0.0
    /// The latest frame's views through the vision tower.
    public var latestVision = 0.0
    public var tokenize = 0.0
    /// The language model over the whole prompt, to the first answer token.
    public var prefill = 0.0
    /// The answer tokens after the first.
    public var decode = 0.0
    public var total = 0.0
    public var timeToFirstToken = 0.0
    public var promptTokens = 0
    public var visualTokens = 0
    public var generatedTokens = 0
    public var candidates = 0

    /// Prompt tokens per second of prefill.
    public var prefillTokensPerSecond: Double { prefill > 0 ? Double(promptTokens) / prefill : 0 }
    /// Answer tokens per second while decoding (the first token comes with the prefill).
    public var decodeTokensPerSecond: Double {
        decode > 0 && generatedTokens > 1 ? Double(generatedTokens - 1) / decode : 0
    }
    /// Answer tokens per second from the question (what a user sees).
    public var endToEndTokensPerSecond: Double { total > 0 ? Double(generatedTokens) / total : 0 }

    public init() {}
}

/// Where the time of one stored frame goes, in seconds.
public struct StoreTimings: Codable, Sendable {
    /// Resize to store size + JPEG round trip.
    public var prep = 0.0
    /// MobileCLIP2-S0 image embedding.
    public var embed = 0.0
    /// Thumbnail through the MiniCPM vision tower.
    public var thumbnail = 0.0
    public var total: Double { prep + embed + thumbnail }
    public init() {}
}

/// When stored frames get their MobileCLIP2-S0 embedding and thumbnail features.
public enum LatentMode: String, Codable, Sendable {
    /// When the frame enters the store, as the video unrolls (default).
    case atStore
    /// At question time, for every candidate frame (the cost a question would pay without the store-time work).
    case atQuestion
}

/// The two models, loaded once.
public final class EfficientStream: @unchecked Sendable {
    public let config: StreamConfig
    public let answerer: Answerer
    public let index: FrameIndex

    /// `directory` is the exported `models` folder (minicpm/, mobileclip/, efficientstream_config.json).
    public init(directory: URL, cacheDirectory: URL) async throws {
        let cfgURL = directory.appendingPathComponent("efficientstream_config.json")
        config = FileManager.default.fileExists(atPath: cfgURL.path) ? try StreamConfig.load(cfgURL) : StreamConfig()
        // the language model runs in bfloat16 (as evaluated); keep MLX's buffer cache modest on a phone
        MLX.GPU.set(cacheLimit: 512 * 1024 * 1024)
        answerer = try await Answerer(directory: directory.appendingPathComponent("minicpm"))
        index = try FrameIndex(
            directory: directory.appendingPathComponent("mobileclip"), cacheDirectory: cacheDirectory,
            roundFp16: config.roundEmbeddingsFp16)
    }

    /// A new stream. `config` overrides the exported settings (e.g. another `gridFps`).
    public func session(config: StreamConfig? = nil, latents: LatentMode = .atStore) -> StreamSession {
        StreamSession(self, config: config ?? self.config, latents: latents)
    }

    /// True when every file the app needs is in `directory`.
    public static func modelsPresent(in directory: URL) -> Bool {
        let fm = FileManager.default
        return [
            "minicpm/model.safetensors", "minicpm/config.json", "minicpm/prompt.json", "minicpm/tokenizer.json",
            "mobileclip/ImageEncoder.mlpackage", "mobileclip/TextEncoder.mlpackage", "mobileclip/vocab.json",
            "mobileclip/merges.txt", "mobileclip/config.json",
        ].allSatisfy { fm.fileExists(atPath: directory.appendingPathComponent($0).path) }
    }
}

public final class StreamSession: @unchecked Sendable {
    /// Prints a timing breakdown of each question (environment variable ES_PROFILE=1).
    static let profile = ProcessInfo.processInfo.environment["ES_PROFILE"] != nil
    public let es: EfficientStream
    private var _config: StreamConfig
    public let latents: LatentMode
    /// The settings of this stream. Changing `gridFps` applies from the next stored frame.
    public var config: StreamConfig {
        get { lock.withLock { _config } }
        set { lock.withLock { _config = newValue } }
    }
    /// The last stored frame's timing.
    public var lastStoreTimings: StoreTimings { lock.withLock { _lastStore } }
    private var _lastStore = StoreTimings()

    private let lock = NSLock()
    private var store: BoundedStore
    private var latest: (time: Double, image: RGBImage)?
    private var nextGrid = 0.0
    /// Bumped by `reset`, so a frame still being embedded for the previous stream is dropped.
    private var generation = 0
    /// All model work (MLX and Core ML) runs here, one call at a time.
    private let work = DispatchQueue(label: "efficientstream.model", qos: .userInitiated)

    init(_ es: EfficientStream, config: StreamConfig, latents: LatentMode) {
        self.es = es
        _config = config
        self.latents = latents
        store = BoundedStore(budget: config.storeBudget)
    }

    // MARK: the stream

    /// Whether `addFrame(t:)` would store a frame at time `t` (lets callers convert only those frames).
    public func wantsStoreFrame(at t: Double) -> Bool {
        lock.withLock { t + 1e-6 >= nextGrid }
    }

    /// A decoded frame at time `t` seconds, in time order. Every frame becomes the latest frame; one frame every
    /// 1 / grid_fps seconds also enters the store (embedded on the model queue; returns when stored).
    public func addFrame(t: Double, image: RGBImage) async throws {
        setLatest(t: t, image: image)
        let store: Bool = lock.withLock {
            guard t + 1e-6 >= nextGrid else { return false }
            while nextGrid <= t + 1e-6 { nextGrid += 1.0 / _config.gridFps }
            return true
        }
        if store { try await addStoreFrame(t: t, image: image) }
    }

    /// Only the latest frame: shown at high resolution as "(now)" when a question is asked. An older frame
    /// (e.g. a store frame finishing after a newer frame was set) does not replace a newer one.
    public func setLatest(t: Double, image: RGBImage) {
        lock.withLock {
            if let l = latest, t < l.time { return }
            latest = (t, image)
        }
    }

    /// Only the store: resize, JPEG round trip, then (`latents == .atStore`) embed and thumbnail features, store.
    @discardableResult
    public func addStoreFrame(t: Double, image: RGBImage, resize: Bool = true) async throws -> StoreTimings {
        let cfg = config
        let es = self.es
        let eager = latents == .atStore
        let gen = lock.withLock { generation }
        let (frame, timings): (StoredFrame, StoreTimings) = try await withCheckedThrowingContinuation { cont in
            work.async {
                do {
                    var tm = StoreTimings()
                    var clock = Date()
                    func lap() -> Double { let n = Date(); defer { clock = n }; return n.timeIntervalSince(clock) }
                    var im = image
                    if resize {
                        im = im.resizedForStore(maxPixels: cfg.gridMaxPixels).jpegRoundTrip(quality: cfg.storeJpegQuality)
                    }
                    tm.prep = lap()
                    var emb: [Float] = []
                    var thumb: EncodedImage?
                    if eager {
                        emb = try es.index.embed(image: im)
                        tm.embed = lap()
                        thumb = es.answerer.encode(
                            im.letterboxed(side: cfg.thumbSide), scaleResolution: cfg.thumbSide, maxSlices: 1)
                        MLX.GPU.clearCache()
                        tm.thumbnail = lap()
                    }
                    cont.resume(returning: (StoredFrame(time: t, image: im, embedding: emb, thumbnail: thumb), tm))
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
        try lock.withLock {
            guard gen == generation else { return }
            _ = try store.add(frame)
            _lastStore = timings
        }
        return timings
    }

    public var storedCount: Int { lock.withLock { store.count } }
    public var storedTimes: [Double] { lock.withLock { store.frames.map(\.time) } }
    public var latestTime: Double? { lock.withLock { latest?.time } }

    /// The stored frames at `times` (for showing the moments an answer used).
    public func storedImages(at times: [Double]) -> [RGBImage] {
        lock.withLock {
            times.compactMap { t in store.frames.first { abs($0.time - t) < 1e-6 }?.image }
        }
    }

    // MARK: questions

    public enum AskError: Error, LocalizedError {
        case noFrame
        public var errorDescription: String? { "No video frame yet." }
    }

    /// Answer at the current moment (the time of the latest frame). `options` = ["A. ...", "B. ...", ...] for
    /// multiple choice, nil for an open-ended answer. `onText` streams the open-ended answer.
    public func ask(_ question: String, options: [String]? = nil, onText: (@Sendable (String) -> Void)? = nil)
        async throws -> StreamResult
    {
        guard let (now, latestImage) = lock.withLock({ latest }) else { throw AskError.noFrame }
        let candidates = lock.withLock { store.before(now) }
        let es = self.es
        let cfg = config
        return try await withCheckedThrowingContinuation { cont in
            work.async {
                do {
                    let start = Date()
                    var tm = QuestionTimings()
                    var clock = start
                    func lap() -> Double { let n = Date(); defer { clock = n }; return n.timeIntervalSince(clock) }
                    let q = es.answerer.sanitize(question)
                    let opts = options?.map { es.answerer.sanitize($0) }
                    var picked: [StoredFrame] = []
                    tm.candidates = candidates.count
                    if !candidates.isEmpty && cfg.retrieveK > 0 {
                        let qe = try es.index.embed(text: q)
                        tm.textEmbed = lap()
                        let embeddings = try candidates.map { f in
                            f.embedding.isEmpty ? try es.index.embed(image: f.image) : f.embedding
                        }
                        tm.frameEmbeds = lap()
                        picked = topKInTimeOrder(embeddings, query: qe, k: cfg.retrieveK).map { candidates[$0] }
                        tm.topK = lap()
                    }
                    var parts: [PromptPart] = []
                    var images: [EncodedImage] = []
                    for f in picked {
                        let thumb = f.thumbnail ?? es.answerer.encode(
                            f.image.letterboxed(side: cfg.thumbSide), scaleResolution: cfg.thumbSide, maxSlices: 1)
                        parts += [.text("frame time[\(formatTime(f.time, style: cfg.timestamps))]: "),
                                  .image(placeholder: thumb.placeholder), .text("\n")]
                        images.append(thumb)
                    }
                    tm.thumbnails = lap()
                    let hi = latestImage.fittingLongSide(cfg.latestLongSide).jpegRoundTrip(quality: cfg.latestJpegQuality)
                    tm.latestPrep = lap()
                    let latestEnc = es.answerer.encode(
                        hi, scaleResolution: cfg.thumbSide, maxSlices: cfg.latestMaxSlices)
                    tm.latestVision = lap()
                    parts += [.text("frame time[\(formatTime(now, style: cfg.timestamps))] (now): "),
                              .image(placeholder: latestEnc.placeholder), .text("\n"),
                              .text(PromptText.questionBlock(q, options: opts))]
                    images.append(latestEnc)
                    let beforeAnswer = Date()
                    let a = es.answerer.answer(
                        parts: parts, images: images, numOptions: opts?.count ?? 0, maxNewTokens: cfg.maxNewTokens,
                        onText: onText.map { f in { text in f(text) } })
                    MLX.GPU.clearCache()
                    let end = Date()
                    tm.tokenize = a.tokenizeSeconds
                    tm.prefill = a.prefillSeconds
                    tm.decode = a.decodeSeconds
                    tm.total = end.timeIntervalSince(start)
                    tm.timeToFirstToken = beforeAnswer.timeIntervalSince(start) + a.tokenizeSeconds + a.prefillSeconds
                    tm.promptTokens = a.promptTokens
                    tm.visualTokens = a.visualTokens
                    tm.generatedTokens = a.generatedTokens
                    if StreamSession.profile {
                        print(String(format: "text %.3f | frames %.3f | topk %.3f | thumbs %.3f | latest prep %.3f | latest vision %.3f | tokenize %.3f | prefill %.3f (%d tok) | decode %.3f (%d tok, %.1f tok/s) | total %.3f",
                                     tm.textEmbed, tm.frameEmbeds, tm.topK, tm.thumbnails, tm.latestPrep, tm.latestVision,
                                     tm.tokenize, tm.prefill, tm.promptTokens, tm.decode, tm.generatedTokens,
                                     tm.decodeTokensPerSecond, tm.total))
                    }
                    cont.resume(
                        returning: StreamResult(
                            text: a.text, probs: a.probs, frameTimes: picked.map(\.time), now: now,
                            promptTokens: a.promptTokens, seconds: tm.total,
                            secondsToFirstToken: tm.timeToFirstToken, timings: tm))
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    /// What a question at `now` would pay if stored frames got their latents only at question time:
    /// MobileCLIP2-S0 over every stored frame older than `now`, and the thumbnails at `thumbnailTimes` through
    /// the vision tower. The results are discarded (measurement only).
    public func timeLatentsAtQuestion(now: Double, thumbnailTimes: [Double]) async throws
        -> (embeds: Double, thumbnails: Double, frames: Int)
    {
        let candidates = lock.withLock { store.before(now) }
        let picked = candidates.filter { f in thumbnailTimes.contains { abs($0 - f.time) < 1e-6 } }
        let es = self.es
        let cfg = config
        return try await withCheckedThrowingContinuation { cont in
            work.async {
                do {
                    let t0 = Date()
                    for f in candidates { _ = try es.index.embed(image: f.image) }
                    let t1 = Date()
                    for f in picked {
                        _ = es.answerer.encode(
                            f.image.letterboxed(side: cfg.thumbSide), scaleResolution: cfg.thumbSide, maxSlices: 1)
                    }
                    MLX.GPU.clearCache()
                    let t2 = Date()
                    cont.resume(returning: (t1.timeIntervalSince(t0), t2.timeIntervalSince(t1), candidates.count))
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    /// Clears the store and the latest frame (a new stream).
    public func reset() {
        lock.withLock {
            store = BoundedStore(budget: _config.storeBudget)
            latest = nil
            nextGrid = 0
            generation += 1
        }
    }
}
