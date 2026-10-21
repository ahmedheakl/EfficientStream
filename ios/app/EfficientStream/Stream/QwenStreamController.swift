import CoreVideo
import EfficientStreamKit
import Foundation
import UIKit

/// One question to the streaming Qwen3.5-0.8B and its (streaming) answer.
@Observable
final class QwenQA: Identifiable {
    let id = UUID()
    let question: String
    let options: [String]?
    let askedAt: Double
    var answer = ""
    var probs: [Double]?
    var result: QwenAnswer?
    var frames = 0
    /// Caption-memory events the question read (PHCM; 0 without the caption memory).
    var memoryEvents = 0
    /// Asked on the caption stream's cache: the seconds it held and its tokens.
    var context: (from: Int, through: Int, tokens: Int)?
    /// Of those, the newest seconds the stream had not taken in yet (prefilled with the question).
    var behind = 0
    /// Asked on the watch's stream (the caption memory as text) rather than the caption stream.
    var onWatch = false
    /// Question -> full answer.
    var seconds = 0.0
    var error: String?
    var isAnswering: Bool { result == nil && error == nil }

    init(question: String, options: [String]?, askedAt: Double) {
        self.question = question
        self.options = options
        self.askedAt = askedAt
    }
}

/// A </Response> of the proactive mode.
struct ProactiveEvent: Identifiable {
    let id = UUID()
    let second: Int
    let text: String
    let seconds: Double
}

/// Feeds the active source to the streaming Qwen3.5-0.8B. Every second's frames (`fps` of them, 4 as trained) go
/// through the vision tower once and on to the streams that watch: the proactive mode's streamer while watching for
/// something, and (PHCM, run-4 weights) the caption stream, which writes </Observe> details and </Summary> segment
/// summaries while the video plays. A question is asked on the caption stream's own KV cache (the last 32 to 96
/// seconds of frames and captions, already prefilled) plus, as text, the captions of the seconds no longer in it;
/// only the question's tokens are prefilled. Without the caption stream, a question looks at the last 16 seconds
/// (one frame per second through the vision tower as it arrives).
@Observable
@MainActor
final class QwenStreamController {
    let qwen: StreamQwen
    let camera = CameraSource()
    let video = VideoFileSource()
    let recent = 16
    let size: QwenFrameSize = .padded(maxPixels: 640 * 360)
    /// Frames per second the streams see (`-esfps`; an even number: the model takes them in two-frame blocks).
    let fps = QwenStreamController.launchFPS
    nonisolated static let launchFPS: Int = {
        let args = ProcessInfo.processInfo.arguments
        guard let i = args.firstIndex(of: "-esfps"), i + 1 < args.count, let n = Int(args[i + 1]) else { return 8 }
        return max(2, n / 2 * 2)
    }()
    /// The caption memory runs (the weights have its prompts: run 4 on).
    let captions: Bool

    private(set) var source: StreamController.Source = .camera
    private(set) var history: [QwenQA] = []
    private(set) var now: Double = 0
    private(set) var framesInWindow = 0
    /// The caption memory: its events (newest last) and last update.
    private(set) var memoryEvents: [CaptionMemory.Event] = []
    private(set) var lastCaption: QwenStreamUpdate?
    private(set) var skippedCaptions = 0
    /// Called after every caption update (demo logging).
    @ObservationIgnored var onCaption: ((QwenStreamUpdate) -> Void)?
    private(set) var isAnswering = false
    /// Resize + patches and vision tower of the last frame.
    private(set) var lastFrame = (prep: 0.0, vision: 0.0)
    private(set) var memoryMB: Double = 0
    // proactive mode
    private(set) var watchPrompt: String?
    private(set) var lastUpdate: QwenStreamUpdate?
    private(set) var events: [ProactiveEvent] = []
    /// Seconds skipped because the previous update was still running.
    private(set) var skippedSeconds = 0
    /// Demo mode (QwenDemoModel): the watch prompt as typed so far (mirrored into the watch field) and start requests,
    /// and the same for a question in the question bar.
    var demoWatchTyping: String?
    private(set) var demoWatchStarts = 0
    var demoTyping: String?
    private(set) var demoSends = 0
    /// Called after every proactive update (demo logging).
    @ObservationIgnored var onUpdate: ((QwenStreamUpdate) -> Void)?

    @ObservationIgnored private let window = FrameWindow()
    @ObservationIgnored private let memory = CaptionMemory()
    @ObservationIgnored private var captioner: QwenStreamer?
    @ObservationIgnored private var captionChain: Task<Void, Never>?
    @ObservationIgnored private let captioning = PendingCount(limit: 4)
    /// The watch and questions go first: the caption stream waits between its answer tokens while they run.
    @ObservationIgnored private let gate = GPUGate()
    @ObservationIgnored private let sampler = Sampler(fps: QwenStreamController.launchFPS)
    /// Each second through the vision tower, in order, before the streams.
    @ObservationIgnored private var secondChain: Task<Void, Never>?
    /// The latest seconds through the vision tower (a question adds those the caption stream has not taken in yet).
    @ObservationIgnored private var recentSeconds: [QwenSecond] = []
    @ObservationIgnored private var encodeChain: Task<Void, Never>?
    @ObservationIgnored private var streamer: QwenStreamer?
    /// Updates run one at a time, in order (they extend the same caches); at most one waits.
    @ObservationIgnored private var updateChain: Task<Void, Never>?
    @ObservationIgnored private let updating = PendingCount()

    init(qwen: StreamQwen) {
        self.qwen = qwen
        // `-esnocaptions`: the watch alone (to time it without the caption stream sharing the GPU)
        captions = qwen.spec.captionSystem != nil && !ProcessInfo.processInfo.arguments.contains("-esnocaptions")
        camera.onFrame = { [weak self] t, pb in self?.frame(t: t, pb: pb, rotation: 0) }
        video.onFrame = { [weak self] t, pb, rot in self?.frame(t: t, pb: pb, rotation: rot) }
        video.onRestart = { [weak self] in Task { @MainActor in self?.resetStream() } }
    }

    // MARK: sources

    func use(_ s: StreamController.Source) {
        guard s != source else { return }
        camera.stop()
        video.pause()
        source = s
        resetStream()
        if s == .camera { camera.start() }
    }

    func start() { if source == .camera { camera.start() } }
    func stop() { camera.stop(); video.pause() }

    func openVideo(_ url: URL, play: Bool = true) async {
        use(.video)
        resetStream()
        await video.load(url)
        if play { video.play() }
    }

    /// Demo mode: starts watching with the prompt in the watch field, as its Start button does.
    func demoStartWatch() { demoWatchStarts += 1 }

    /// Demo mode: sends the question in the question bar, as the send button does.
    func demoSend() { demoSends += 1 }

    func resetStream() {
        window.clear()
        memory.clear()
        memoryEvents = []
        lastCaption = nil
        skippedCaptions = 0
        captionChain = nil
        secondChain = nil
        recentSeconds = []
        captioner = captions ? QwenStreamer(qwen: qwen, prompt: nil, context: .evict(8)) : nil
        captioner?.beforeToken = { [gate] in gate.wait() }
        sampler.reset()
        sampler.resetQuarters(watching: watchPrompt != nil || captions)
        encodeChain?.cancel()
        encodeChain = nil
        now = 0
        framesInWindow = 0
        if let p = watchPrompt { watch(p) }
    }

    func clearHistory() {
        history.removeAll()
        events.removeAll()
    }

    // MARK: proactive mode

    func watch(_ prompt: String?) {
        updateChain = nil
        watchPrompt = prompt
        // the runner's window recomputed every second is not real time on the phone: keep appending, and every 8
        // seconds drop the oldest seconds from the attention caches (the first turn and the last 32 seconds stay)
        streamer = prompt.map { QwenStreamer(qwen: qwen, prompt: $0, context: .evict(8), noRepeat: 4) }
        lastUpdate = nil
        skippedSeconds = 0
        sampler.resetQuarters(watching: prompt != nil || captions)
    }

    // MARK: frames

    /// Called on the source's queue for every decoded frame.
    nonisolated private func frame(t: Double, pb: CVPixelBuffer, rotation: Int) {
        let s = sampler.offer(t)
        Task { @MainActor in self.now = t }
        if s.newSecond && !captions {
            let image = RGBImage(pixelBuffer: pb, rotation: rotation)
            Task { @MainActor in self.encode(image, t: t) }
        }
        if let done = s.completedSecond {
            Task { @MainActor in self.encodeSecond(second: done.second, frames: done.frames, times: done.times) }
        }
        if s.quarter {  // only while watching
            sampler.addQuarter(RGBImage(pixelBuffer: pb, rotation: rotation), t: t)
        }
    }

    private func encode(_ image: RGBImage, t: Double) {
        let previous = encodeChain
        let (qwen, size, window, recent) = (self.qwen, self.size, self.window, self.recent)
        encodeChain = Task {
            await previous?.value
            let f = await Task.detached(priority: .userInitiated) { qwen.encode(image, time: t, size: size) }.value
            window.append(f, keep: recent)
            self.framesInWindow = window.count
            self.lastFrame = (f.prepSeconds, f.visionSeconds)
        }
    }

    /// One second through the vision tower once, then to the watch and the caption stream.
    private func encodeSecond(second: Int, frames: [RGBImage], times: [Double]) {
        guard streamer != nil || captioner != nil else { return }
        let (qwen, size, previous) = (self.qwen, QwenFrameSize.resized(minPixels: qwen.spec.minPixels,
                                                                       maxPixels: qwen.spec.maxPixels), secondChain)
        secondChain = Task {
            await previous?.value
            let sec = await Task.detached(priority: .userInitiated) {
                QwenSecond(qwen: qwen, size: size, second: second, frames: frames, times: times)
            }.value
            self.lastFrame = (sec.prep / Double(max(1, sec.frames)), sec.vision / Double(max(1, sec.frames)))
            self.recentSeconds = Array((self.recentSeconds + [sec]).suffix(16))
            self.update(sec)
            self.caption(sec)
        }
    }

    /// One second through the caption stream, in order (a backlog of at most 4 seconds; beyond, seconds are skipped).
    private func caption(_ sec: QwenSecond) {
        guard let captioner else { return }
        guard captioning.tryAcquire() else {
            skippedCaptions += 1
            return
        }
        let (captioning, memory, previous) = (self.captioning, self.memory, captionChain)
        captionChain = Task {
            await previous?.value
            let u = await Task.detached(priority: .userInitiated) {
                let u = captioner.update(sec)
                memory.add(u.answer, second: Double(sec.second), end: Double(sec.second + 1))
                return u
            }.value
            captioning.release()
            guard captioner === self.captioner else { return }
            print(String(format: "qwen caption %ds: %.3f s (prefill %d tok %.3f, answer %d tok %.3f) | %@", u.second,
                         u.total, u.prefillTokens, u.prefill, u.decodeTokens, u.decode, u.answer))
            self.lastCaption = u
            self.memoryEvents = memory.all
            self.onCaption?(u)
        }
    }

    private func update(_ sec: QwenSecond) {
        guard let streamer, watchPrompt != nil else { return }
        guard updating.tryAcquire() else {
            skippedSeconds += 1
            return
        }
        let (updating, gate) = (self.updating, self.gate)
        let previous = updateChain
        updateChain = Task {
            await previous?.value
            let u = await Task.detached(priority: .userInitiated) {
                gate.begin()
                defer { gate.end() }
                return streamer.update(sec)
            }.value
            updating.release()
            guard streamer === self.streamer else { return }
            print(String(format: "qwen watch %ds: %.3f s (vision %.3f, prefill %d tok %.3f, answer %d tok %.3f), "
                             + "%d s skipped | %@", u.second, u.total, u.vision, u.prefillTokens, u.prefill,
                         u.decodeTokens, u.decode, self.skippedSeconds, u.answer))
            self.lastUpdate = u
            self.onUpdate?(u)
            if u.answer.hasPrefix("</Response>") {
                let text = u.answer.dropFirst("</Response>".count).trimmingCharacters(in: .whitespaces)
                self.events.insert(ProactiveEvent(second: sec.second, text: text, seconds: u.total), at: 0)
            }
            self.memoryMB = DeviceStats.memoryMB().current
        }
    }

    /// Command-line checks of the live screen: `-eswatch <prompt>` starts the proactive mode at launch,
    /// `-esask <question>` asks after 20 seconds of stream.
    func runLaunchChecks() async {
        let args = ProcessInfo.processInfo.arguments
        if let i = args.firstIndex(of: "-eswatch"), i + 1 < args.count { watch(args[i + 1]) }
        if let i = args.firstIndex(of: "-esask"), i + 1 < args.count {
            try? await Task.sleep(for: .seconds(20))
            ask(args[i + 1], options: nil)
        }
    }

    // MARK: questions

    /// Asks at the current moment. With the caption stream: on its KV cache (`QwenStreamer.ask`), the captions of
    /// the seconds no longer in it given as text; otherwise the last 16 seconds (PHCM's prompt without memory).
    /// `options` (multiple choice) are lines like "A. a cup" (the last 16 seconds only).
    func ask(_ question: String, options: [String]?) {
        let q = question.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty else { return }
        // while watching, the watch's stream is the up-to-date one (the caption stream may run behind): ask there,
        // with the whole caption memory as text
        if options == nil, let streamer, lastUpdate != nil {
            askStream(q, streamer, allCaptions: true)
            return
        }
        if options == nil, let captioner, lastCaption != nil {
            askStream(q, captioner, allCaptions: false)
            return
        }
        let latest = window.snapshot()
        guard !latest.isEmpty else { return }
        let qa = QwenQA(question: q, options: options, askedAt: now)
        history.insert(qa, at: 0)
        isAnswering = true
        let qwen = self.qwen
        Task {
            let t0 = Date()
            let a = await Task.detached(priority: .userInitiated) {
                qwen.ask(frames: latest, question: q, options: options, memory: nil) { text in
                    Task { @MainActor in qa.answer = text }
                }
            }.value
            qa.seconds = Date().timeIntervalSince(t0)
            qa.answer = a.text
            qa.probs = a.probs
            qa.frames = latest.count
            qa.result = a
            self.isAnswering = false
            self.memoryMB = DeviceStats.memoryMB().current
            print(String(format: "qwen question: %d frames, first words %.3f s, total %.3f s, %d prompt tokens, %d tokens | %@",
                         latest.count, qa.seconds - a.decodeSeconds, qa.seconds, a.promptTokens, a.generatedTokens, a.text))
        }
    }

    private func askStream(_ q: String, _ stream: QwenStreamer, allCaptions: Bool) {
        let qa = QwenQA(question: q, options: nil, askedAt: now)
        history.insert(qa, at: 0)
        isAnswering = true
        let (memory, spec, gate, recent) = (self.memory, qwen.spec, self.gate, recentSeconds)
        Task {
            let t0 = Date()
            let out = await Task.detached(priority: .userInitiated) { () -> (QwenAnswer, QwenStreamer.Snapshot, Int, Int)? in
                // the stream's context now (once the update running ends), and the captions of the seconds it no
                // longer holds
                guard let snap = stream.snapshot() else { return nil }
                let older = allCaptions ? memory.all : memory.all.filter { $0.end <= Double(snap.from) }
                let text = older.isEmpty ? nil : CaptionMemory.text(older, preamble: spec.memoryPreamble ?? "",
                                                                     postamble: spec.memoryPostamble ?? "")
                gate.begin()
                defer { gate.end() }
                let r = stream.ask(q, on: snap, memory: text, recent: recent) { t in Task { @MainActor in qa.answer = t } }
                return (r.0, r.1, older.count, r.2)
            }.value
            self.isAnswering = false
            guard let (a, snap, older, behind) = out else {
                qa.error = "no stream yet"
                return
            }
            qa.seconds = Date().timeIntervalSince(t0)
            qa.answer = a.text
            qa.frames = snap.frames + behind * self.fps
            qa.memoryEvents = older
            qa.context = (snap.from, snap.through + behind, snap.contextTokens)
            qa.behind = behind
            qa.onWatch = allCaptions
            qa.result = a
            self.memoryMB = DeviceStats.memoryMB().current
            print(String(format: "qwen question on the " + (allCaptions ? "watch" : "caption") + " stream cache (%d-%ds, %d frames, %d tokens) + %d newer seconds + %d older captions: "
                             + "first words %.3f s, total %.3f s, %d prompt tokens, %d tokens | %@", snap.from,
                         snap.through, snap.frames, snap.contextTokens, behind, older, qa.seconds - a.decodeSeconds, qa.seconds,
                         a.promptTokens, a.generatedTokens, a.raw))
        }
    }
}

/// The last frames through the vision tower, shared between the encoding task and questions.
final class FrameWindow: @unchecked Sendable {
    private let lock = NSLock()
    private var frames: [QwenFrame] = []

    var count: Int { lock.withLock { frames.count } }
    func append(_ f: QwenFrame, keep: Int) {
        lock.withLock {
            frames.append(f)
            if frames.count > keep { frames.removeFirst(frames.count - keep) }
        }
    }
    func snapshot() -> [QwenFrame] { lock.withLock { frames } }
    func clear() { lock.withLock { frames.removeAll() } }
}

/// Which frames to keep: the first frame of each second (questions), and while watching the first frame of each
/// quarter second, grouped by second (the proactive mode's 4 fps, as the runner samples them).
final class Sampler: @unchecked Sendable {
    struct Offer {
        var newSecond = false
        var quarter = false
        var completedSecond: (second: Int, frames: [RGBImage], times: [Double])?
    }

    private let lock = NSLock()
    private var lastSecond = -1
    private var lastQuarter = -1
    private var quarterSecond = -1
    private var frames: [RGBImage] = []
    private var times: [Double] = []
    private var watching = false
    let fps: Int

    init(fps: Int) { self.fps = fps }

    func reset() {
        lock.withLock {
            lastSecond = -1
            lastQuarter = -1
            quarterSecond = -1
            frames = []
            times = []
        }
    }

    func resetQuarters(watching w: Bool) {
        lock.withLock {
            watching = w
            lastQuarter = -1
            quarterSecond = -1
            frames = []
            times = []
        }
    }

    func offer(_ t: Double) -> Offer {
        lock.withLock {
            var o = Offer()
            let s = Int(t.rounded(.down))
            if s > lastSecond {
                lastSecond = s
                o.newSecond = true
            }
            guard watching else { return o }
            let q = Int((t * Double(fps) + 1e-7).rounded(.down))
            if q > lastQuarter {
                lastQuarter = q
                o.quarter = true
                if s > quarterSecond {
                    if quarterSecond >= 0 && !frames.isEmpty { o.completedSecond = (quarterSecond, frames, times) }
                    quarterSecond = s
                    frames = []
                    times = []
                }
            }
            return o
        }
    }

    func addQuarter(_ image: RGBImage, t: Double) {
        lock.withLock {
            frames.append(image)
            times.append(t)
        }
    }
}

/// Lets urgent work (the watch, a question) run before the caption stream's next answer token.
final class GPUGate: @unchecked Sendable {
    private let cond = NSCondition()
    private var urgent = 0

    func begin() { cond.withLock { urgent += 1 } }
    func end() {
        cond.withLock {
            urgent -= 1
            cond.broadcast()
        }
    }

    /// Waits while urgent work runs (at most 5 s).
    func wait() {
        let limit = Date().addingTimeInterval(5)
        cond.withLock {
            while urgent > 0 && Date() < limit { _ = cond.wait(until: limit) }
        }
    }
}
