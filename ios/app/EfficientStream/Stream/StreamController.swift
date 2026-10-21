import CoreVideo
import EfficientStreamKit
import Foundation
import UIKit

/// One question and its (streaming) answer.
@Observable
final class QA: Identifiable {
    let id = UUID()
    let question: String
    let options: [String]?
    let askedAt: Double
    var answer = ""
    var probs: [Double]?
    var result: StreamResult?
    var thumbnails: [(time: Double, image: UIImage)] = []
    var error: String?
    var isAnswering: Bool { result == nil && error == nil }

    init(question: String, options: [String]?, askedAt: Double) {
        self.question = question
        self.options = options
        self.askedAt = askedAt
    }
}

/// Feeds the active source into a StreamSession and runs questions.
@Observable
@MainActor
final class StreamController {
    enum Source: String, CaseIterable, Identifiable {
        case camera = "Camera", video = "Video"
        var id: String { rawValue }
    }

    let session: StreamSession
    let camera = CameraSource()
    let video = VideoFileSource()
    private(set) var source: Source = .camera
    private(set) var history: [QA] = []
    private(set) var now: Double = 0
    private(set) var storedCount = 0
    private(set) var isAnswering = false
    private(set) var lastStoreSeconds: Double = 0
    private(set) var lastStore = StoreTimings()
    private(set) var memoryMB: Double = 0
    /// Frames not stored on time because the previous one was still being indexed.
    private(set) var lateStores = 0
    /// Demo mode (DemoModel): the question as typed so far, mirrored into the question bar, and send requests.
    var demoTyping: String?
    private(set) var demoSends = 0
    /// Seconds between stored frames (2 = the evaluated 0.5 fps).
    var indexInterval: Double = 2 {
        didSet {
            var c = session.config
            c.gridFps = 1 / indexInterval
            session.config = c
            UserDefaults.standard.set(indexInterval, forKey: "indexInterval")
        }
    }

    /// The newest frame (not yet converted): converted only when a question is asked.
    @ObservationIgnored private let latest = LatestFrame()
    /// Store frames are embedded one at a time, in time order.
    @ObservationIgnored private var storeChain: Task<Void, Never>?
    /// Store frames converted and waiting (at most one waits; a later one is skipped and stored late).
    @ObservationIgnored private let pending = PendingCount()

    init(es: EfficientStream) {
        session = es.session()
        let saved = UserDefaults.standard.double(forKey: "indexInterval")
        if saved > 0 { indexInterval = saved }
        camera.onFrame = { [weak self] t, pb in self?.frame(t: t, pb: pb, rotation: 0) }
        video.onFrame = { [weak self] t, pb, rot in self?.frame(t: t, pb: pb, rotation: rot) }
        video.onRestart = { [weak self] in Task { @MainActor in self?.resetStream() } }
    }

    // MARK: sources

    func use(_ s: Source) {
        guard s != source else { return }
        camera.stop()
        video.pause()
        source = s
        resetStream()
        if s == .camera { camera.start() }
    }

    func start() { if source == .camera { camera.start() } }
    func stop() { camera.stop(); video.pause() }

    func openVideo(_ url: URL, start: Double = 0) async {
        use(.video)
        resetStream()
        await video.load(url)
        if start > 0 { await video.seek(to: start) }
        video.play()
    }

    /// Demo mode: sends the question in the question bar, as the send button does.
    func demoSend() { demoSends += 1 }

    func resetStream() {
        session.reset()
        latest.clear()
        storeChain?.cancel()
        storeChain = nil
        now = 0
        storedCount = 0
        lateStores = 0
    }

    func clearHistory() { history.removeAll() }

    /// Called on the source's queue for every decoded frame.
    nonisolated private func frame(t: Double, pb: CVPixelBuffer, rotation: Int) {
        latest.set(t: t, pb: pb, rotation: rotation)
        guard session.wantsStoreFrame(at: t) else {
            Task { @MainActor in self.now = t }
            return
        }
        // indexing slower than the interval: keep one frame waiting, skip the rest (the next one is stored late)
        guard pending.tryAcquire() else {
            Task { @MainActor in self.now = t; self.lateStores += 1 }
            return
        }
        let image = RGBImage(pixelBuffer: pb, rotation: rotation)
        let pending = self.pending
        Task { @MainActor in
            self.now = t
            let previous = self.storeChain
            self.storeChain = Task {
                await previous?.value
                let start = Date()
                do {
                    try await self.session.addFrame(t: t, image: image)
                } catch {
                    print("store frame \(t): \(error)")
                }
                pending.release()
                self.lastStoreSeconds = Date().timeIntervalSince(start)
                self.lastStore = self.session.lastStoreTimings
                self.storedCount = self.session.storedCount
                self.memoryMB = DeviceStats.memoryMB().current
            }
        }
    }

    // MARK: questions

    /// Asks at the current moment. `options` (multiple choice) are lines like "A. a cup".
    func ask(_ question: String, options: [String]?) {
        let q = question.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty, let frame = latest.get() else { return }
        let qa = QA(question: q, options: options, askedAt: frame.t)
        history.insert(qa, at: 0)
        isAnswering = true
        let session = self.session
        let pending = PendingFrame(pb: frame.pb, rotation: frame.rotation)
        Task {
            // the latest frame at full resolution, converted only now (off the main thread)
            let image = await Task.detached(priority: .userInitiated) { pending.image() }.value
            session.setLatest(t: frame.t, image: image)
            do {
                let r = try await session.ask(q, options: options) { text in
                    Task { @MainActor in qa.answer = text }
                }
                let images = session.storedImages(at: r.frameTimes)
                let thumbs: [(Double, UIImage)] = zip(r.frameTimes, images).compactMap { t, im in
                    im.uiImage.map { (t, $0) }
                }
                qa.answer = r.text
                qa.probs = r.probs
                qa.thumbnails = thumbs
                qa.result = r
                memoryMB = DeviceStats.memoryMB().current
                print(String(format: "question: ttft %.3f s, total %.3f s, %d tokens, %.1f tok/s",
                             r.timings.timeToFirstToken, r.timings.total, r.timings.generatedTokens,
                             r.timings.decodeTokensPerSecond))
            } catch {
                qa.error = error.localizedDescription
            }
            isAnswering = false
        }
    }
}

/// How many store frames are waiting to be indexed (at most 2: one indexing, one waiting).
final class PendingCount: @unchecked Sendable {
    private let lock = NSLock()
    private var n = 0
    private let limit: Int
    init(limit: Int = 2) { self.limit = limit }
    func tryAcquire() -> Bool { lock.withLock { if n >= limit { return false }; n += 1; return true } }
    func release() { lock.withLock { n -= 1 } }
}

/// A pixel buffer handed to a background task for conversion.
struct PendingFrame: @unchecked Sendable {
    let pb: CVPixelBuffer
    let rotation: Int
    func image() -> RGBImage { RGBImage(pixelBuffer: pb, rotation: rotation) }
}

/// The newest pixel buffer, shared between the capture queue and the main actor.
final class LatestFrame: @unchecked Sendable {
    private let lock = NSLock()
    private var frame: (t: Double, pb: CVPixelBuffer, rotation: Int)?

    func set(t: Double, pb: CVPixelBuffer, rotation: Int) { lock.withLock { frame = (t, pb, rotation) } }
    func get() -> (t: Double, pb: CVPixelBuffer, rotation: Int)? { lock.withLock { frame } }
    func clear() { lock.withLock { frame = nil } }
}
