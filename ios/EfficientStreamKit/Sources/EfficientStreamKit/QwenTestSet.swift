// The streaming test set (manifest.json + videos/, as TestSetRunner) for the streaming Qwen3.5-0.8B: each video
// is decoded in order, one frame per second is kept (the first frame at or after each second) and goes through the
// vision tower when it arrives, and each question is asked on the most recent frames when the stream reaches its
// time (proactive/stream_qwen.py `ask`: 16 frames). One JSON line per question.

import AVFoundation
import Foundation

public struct QwenTimings: Codable, Sendable {
    /// Frames in the prompt, and the mean per frame over this video so far: resize / pad + patches, vision tower.
    public var frames = 0
    public var framePrep = 0.0
    public var frameVision = 0.0
    /// Question -> answer (the frames were encoded when they arrived): prompt text -> ids, embeddings + prefill to
    /// the first token's logits, the remaining tokens.
    public var tokenize = 0.0
    public var prefill = 0.0
    public var decode = 0.0
    public var timeToFirstToken = 0.0
    public var total = 0.0
    public var promptTokens = 0
    public var visualTokens = 0
    public var generatedTokens = 0
    /// The multiple-choice prompt (the same frames).
    public var mcTotal = 0.0
    public var mcPromptTokens = 0

    public var prefillTokensPerSecond: Double { prefill > 0 ? Double(promptTokens) / prefill : 0 }
    public var decodeTokensPerSecond: Double {
        decode > 0 && generatedTokens > 1 ? Double(generatedTokens - 1) / decode : 0
    }
}

public struct QwenRecord: Codable, Sendable {
    public var id: Int
    public var run: String
    public var model: String
    public var frameSize: String
    public var bench: String
    public var task: String
    public var question: String
    public var reference: String
    public var answer: String
    public var mcAnswer: String?
    public var mcCorrect: Bool?
    public var mcProbs: [Double]?
    public var now: Double
    public var frameTimes: [Double]
    /// Pixels given to the vision tower (the last frame).
    public var framePixels: String
    public var timings: QwenTimings
    public var memoryMB: Double
    public var peakMemoryMB: Double
    public var thermal: String
    public var device: String
}

public final class QwenTestSetRunner: @unchecked Sendable {
    public let qwen: StreamQwen
    public let questions: [TestQuestion]
    public let videos: URL
    public let output: URL

    public init(qwen: StreamQwen, manifest: URL, videos: URL, output: URL) throws {
        self.qwen = qwen
        questions = try JSONDecoder().decode([TestQuestion].self, from: Data(contentsOf: manifest))
        self.videos = videos
        self.output = output
    }

    public func done(run: String) -> Set<Int> {
        guard let text = try? String(contentsOf: output, encoding: .utf8) else { return [] }
        var ids = Set<Int>()
        for line in text.split(separator: "\n") {
            if let r = try? JSONDecoder().decode(QwenRecord.self, from: Data(line.utf8)), r.run == run { ids.insert(r.id) }
        }
        return ids
    }

    public func run(
        _ run: String, size: QwenFrameSize, recent: Int = 16, multipleChoice: Bool = true, limit: Int? = nil,
        benches: Set<String>? = nil, progress: @escaping @Sendable (Int, Int, QwenRecord) -> Void
    ) async throws {
        let skip = done(run: run)
        let selected = questions.filter { benches?.contains($0.bench) ?? true }
        var todo = selected.filter { !skip.contains($0.id) }
        if let limit { todo = Array(todo.prefix(limit)) }
        let byVideo = Dictionary(grouping: todo, by: \.file)
        var count = skip.count
        let total = limit.map { skip.count + min($0, todo.count) } ?? selected.count
        if !FileManager.default.fileExists(atPath: output.path) {
            FileManager.default.createFile(atPath: output.path, contents: nil)
        }
        let handle = try FileHandle(forWritingTo: output)
        defer { try? handle.close() }
        try handle.seekToEnd()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]

        for file in byVideo.keys.sorted() {
            let qs = byVideo[file]!.sorted { ($0.time, $0.id) < ($1.time, $1.id) }
            var window: [QwenFrame] = []
            var prepSum = 0.0, visionSum = 0.0, encoded = 0

            func ask(_ q: TestQuestion, now: Double) throws {
                let frames = Array(window.suffix(recent))
                let t0 = Date()
                let a = qwen.ask(frames: frames, question: q.question, options: nil)
                var tm = QwenTimings()
                tm.frames = frames.count
                tm.framePrep = encoded > 0 ? prepSum / Double(encoded) : 0
                tm.frameVision = encoded > 0 ? visionSum / Double(encoded) : 0
                tm.tokenize = a.tokenizeSeconds
                tm.prefill = a.prefillSeconds
                tm.decode = a.decodeSeconds
                tm.timeToFirstToken = a.tokenizeSeconds + a.prefillSeconds
                tm.total = Date().timeIntervalSince(t0)
                tm.promptTokens = a.promptTokens
                tm.visualTokens = a.visualTokens
                tm.generatedTokens = a.generatedTokens
                var rec = QwenRecord(
                    id: q.id, run: run, model: "stream-qwen3.5-0.8b", frameSize: size.description, bench: q.bench,
                    task: q.task, question: q.question, reference: q.answer, answer: a.text, now: now,
                    frameTimes: frames.map(\.time),
                    framePixels: frames.last.map { "\($0.width)x\($0.height)" } ?? "",
                    timings: tm, memoryMB: 0, peakMemoryMB: 0, thermal: DeviceStats.thermalState,
                    device: DeviceStats.model)
                if multipleChoice, let opts = q.options {
                    let lettered = opts.enumerated().map { "\(Character(UnicodeScalar(65 + $0.offset)!)). \($0.element)" }
                    let m0 = Date()
                    let m = qwen.ask(frames: frames, question: q.question, options: lettered)
                    rec.timings.mcTotal = Date().timeIntervalSince(m0)
                    rec.timings.mcPromptTokens = m.promptTokens
                    rec.mcAnswer = m.text
                    rec.mcProbs = m.probs
                    if let gt = q.gt { rec.mcCorrect = m.text == String(Character(UnicodeScalar(65 + gt)!)) }
                }
                let mem = DeviceStats.memoryMB()
                rec.memoryMB = mem.current
                rec.peakMemoryMB = mem.peak
                try handle.write(contentsOf: encoder.encode(rec) + Data("\n".utf8))
                count += 1
                progress(count, total, rec)
            }

            let asset = AVURLAsset(url: videos.appendingPathComponent(file))
            guard let track = try await asset.loadTracks(withMediaType: .video).first else { continue }
            let reader = try AVAssetReader(asset: asset)
            let out = AVAssetReaderTrackOutput(
                track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
            out.alwaysCopiesSampleData = false
            reader.add(out)
            reader.startReading()
            var qi = 0
            var nextSecond = 0.0
            var lastT = 0.0
            while qi < qs.count, let sb = out.copyNextSampleBuffer() {
                guard let pb = CMSampleBufferGetImageBuffer(sb) else { continue }
                let t = CMSampleBufferGetPresentationTimeStamp(sb).seconds
                while qi < qs.count && t > qs[qi].time + 1e-6 {
                    try ask(qs[qi], now: lastT)
                    qi += 1
                }
                if qi == qs.count { break }
                lastT = t
                guard t + 1e-6 >= nextSecond else { continue }
                nextSecond = t.rounded(.down) + 1
                // only frames that can be in a question's window are encoded (the stream would encode every one)
                guard qs[qi].time - t < Double(recent) + 1 else { continue }
                let f = qwen.encode(RGBImage(pixelBuffer: pb), time: t, size: size)
                prepSum += f.prepSeconds
                visionSum += f.visionSeconds
                encoded += 1
                window.append(f)
                if window.count > recent { window.removeFirst(window.count - recent) }
            }
            reader.cancelReading()
            while qi < qs.count {
                try ask(qs[qi], now: lastT)
                qi += 1
            }
        }
    }
}
