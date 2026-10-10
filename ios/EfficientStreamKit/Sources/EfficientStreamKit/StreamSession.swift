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

    public func session() -> StreamSession { StreamSession(self) }

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
    public let es: EfficientStream
    public var config: StreamConfig { es.config }

    private let lock = NSLock()
    private var store: BoundedStore
    private var latest: (time: Double, image: RGBImage)?
    private var nextGrid = 0.0
    /// Bumped by `reset`, so a frame still being embedded for the previous stream is dropped.
    private var generation = 0
    /// All model work (MLX and Core ML) runs here, one call at a time.
    private let work = DispatchQueue(label: "efficientstream.model", qos: .userInitiated)

    init(_ es: EfficientStream) {
        self.es = es
        store = BoundedStore(budget: es.config.storeBudget)
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
            while nextGrid <= t + 1e-6 { nextGrid += 1.0 / config.gridFps }
            return true
        }
        if store { try await addStoreFrame(t: t, image: image) }
    }

    /// Only the latest frame: shown at high resolution as "(now)" when a question is asked.
    public func setLatest(t: Double, image: RGBImage) {
        lock.withLock { latest = (t, image) }
    }

    /// Only the store: resize, JPEG round trip, embed, thumbnail features, store.
    public func addStoreFrame(t: Double, image: RGBImage, resize: Bool = true) async throws {
        let cfg = config
        let es = self.es
        let gen = lock.withLock { generation }
        let frame: StoredFrame = try await withCheckedThrowingContinuation { cont in
            work.async {
                do {
                    var im = image
                    if resize {
                        im = im.resizedForStore(maxPixels: cfg.gridMaxPixels).jpegRoundTrip(quality: cfg.storeJpegQuality)
                    }
                    let emb = try es.index.embed(image: im)
                    let thumb = es.answerer.encode(
                        im.letterboxed(side: cfg.thumbSide), scaleResolution: cfg.thumbSide, maxSlices: 1)
                    MLX.GPU.clearCache()
                    cont.resume(returning: StoredFrame(time: t, image: im, embedding: emb, thumbnail: thumb))
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
        try lock.withLock {
            guard gen == generation else { return }
            _ = try store.add(frame)
        }
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
                    let q = es.answerer.sanitize(question)
                    let opts = options?.map { es.answerer.sanitize($0) }
                    var picked: [StoredFrame] = []
                    if !candidates.isEmpty && cfg.retrieveK > 0 {
                        let qe = try es.index.embed(text: q)
                        picked = topKInTimeOrder(candidates.map(\.embedding), query: qe, k: cfg.retrieveK)
                            .map { candidates[$0] }
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
                    let hi = latestImage.fittingLongSide(cfg.latestLongSide).jpegRoundTrip(quality: cfg.latestJpegQuality)
                    let latestEnc = es.answerer.encode(
                        hi, scaleResolution: cfg.thumbSide, maxSlices: cfg.latestMaxSlices)
                    parts += [.text("frame time[\(formatTime(now, style: cfg.timestamps))] (now): "),
                              .image(placeholder: latestEnc.placeholder), .text("\n"),
                              .text(PromptText.questionBlock(q, options: opts))]
                    images.append(latestEnc)
                    var firstToken: Date?
                    let a = es.answerer.answer(
                        parts: parts, images: images, numOptions: opts?.count ?? 0, maxNewTokens: cfg.maxNewTokens
                    ) { text in
                        if firstToken == nil { firstToken = Date() }
                        onText?(text)
                    }
                    MLX.GPU.clearCache()
                    let end = Date()
                    cont.resume(
                        returning: StreamResult(
                            text: a.text, probs: a.probs, frameTimes: picked.map(\.time), now: now,
                            promptTokens: a.promptTokens, seconds: end.timeIntervalSince(start),
                            secondsToFirstToken: (firstToken ?? end).timeIntervalSince(start)))
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    /// Clears the store and the latest frame (a new stream).
    public func reset() {
        lock.withLock {
            store = BoundedStore(budget: config.storeBudget)
            latest = nil
            nextGrid = 0
            generation += 1
        }
    }
}
