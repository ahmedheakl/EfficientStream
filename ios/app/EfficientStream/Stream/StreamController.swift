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

    /// The newest frame (not yet converted): converted only when a question is asked.
    @ObservationIgnored private let latest = LatestFrame()
    /// Store frames are embedded one at a time, in time order.
    @ObservationIgnored private var storeChain: Task<Void, Never>?

    init(es: EfficientStream) {
        session = es.session()
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

    func openVideo(_ url: URL) async {
        use(.video)
        resetStream()
        await video.load(url)
        video.play()
    }

    func resetStream() {
        session.reset()
        latest.clear()
        storeChain?.cancel()
        storeChain = nil
        now = 0
        storedCount = 0
    }

    func clearHistory() { history.removeAll() }

    /// Called on the source's queue for every decoded frame.
    nonisolated private func frame(t: Double, pb: CVPixelBuffer, rotation: Int) {
        latest.set(t: t, pb: pb, rotation: rotation)
        guard session.wantsStoreFrame(at: t) else {
            Task { @MainActor in self.now = t }
            return
        }
        let image = RGBImage(pixelBuffer: pb, rotation: rotation)
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
                self.lastStoreSeconds = Date().timeIntervalSince(start)
                self.storedCount = self.session.storedCount
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
            } catch {
                qa.error = error.localizedDescription
            }
            isAnswering = false
        }
    }
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
