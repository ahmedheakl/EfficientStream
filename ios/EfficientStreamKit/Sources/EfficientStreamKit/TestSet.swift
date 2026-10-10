// Runs a streaming test set (tools: es-bench manifest.json + videos/): each video is decoded in order, frames enter
// the session as they would from a player, and each question is asked when the stream reaches its time (the latest
// frame is the last frame at or before that time). One JSON line per question, with the answer and the timings.

import AVFoundation
import Foundation

public struct TestQuestion: Codable, Sendable {
    public var id: Int
    public var bench: String
    public var task: String
    public var video: String
    public var time: Double
    public var question: String
    public var answer: String
    public var options: [String]?
    public var gt: Int?
    public var file: String
}

public struct TestRecord: Codable, Sendable {
    public var id: Int
    public var run: String
    public var bench: String
    public var task: String
    public var question: String
    public var reference: String
    public var answer: String
    public var timings: QuestionTimings
    /// Multiple choice on the same question (OVO-Bench, StreamingBench): letter, correct, option probabilities.
    public var mcAnswer: String?
    public var mcCorrect: Bool?
    public var mcProbs: [Double]?
    public var mcSeconds: Double?
    /// Measured only: the same question with the stored frames' latents computed at question time.
    public var latentsAtQuestionEmbeds: Double?
    public var latentsAtQuestionThumbnails: Double?
    public var gridFps: Double
    public var stored: Int
    public var frameTimes: [Double]
    public var now: Double
    /// Mean time per stored frame in this video so far, and its parts.
    public var storeMean: StoreTimings
    public var storeFrames: Int
    public var memoryMB: Double
    public var peakMemoryMB: Double
    public var thermal: String
    public var device: String
}

public final class TestSetRunner: @unchecked Sendable {
    public let es: EfficientStream
    public let questions: [TestQuestion]
    public let videos: URL
    public let output: URL

    public init(es: EfficientStream, manifest: URL, videos: URL, output: URL) throws {
        self.es = es
        questions = try JSONDecoder().decode([TestQuestion].self, from: Data(contentsOf: manifest))
        self.videos = videos
        self.output = output
    }

    /// Ids already in the output for `run` (a run can be resumed).
    public func done(run: String) -> Set<Int> {
        guard let text = try? String(contentsOf: output, encoding: .utf8) else { return [] }
        var ids = Set<Int>()
        for line in text.split(separator: "\n") {
            if let r = try? JSONDecoder().decode(TestRecord.self, from: Data(line.utf8)), r.run == run { ids.insert(r.id) }
        }
        return ids
    }

    /// Runs every question not yet in the output for `run`. `progress(done, total, record)` after each question.
    public func run(
        _ run: String, gridFps: Double, measureLatentsAtQuestion: Bool, multipleChoice: Bool = true,
        limit: Int? = nil, progress: @escaping @Sendable (Int, Int, TestRecord) -> Void
    ) async throws {
        let skip = done(run: run)
        var todo = questions.filter { !skip.contains($0.id) }
        if let limit { todo = Array(todo.prefix(limit)) }
        let byVideo = Dictionary(grouping: todo, by: \.file)
        var config = es.config
        config.gridFps = gridFps
        var count = skip.count
        let total = limit.map { skip.count + min($0, todo.count) } ?? questions.count
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
            let session = es.session(config: config)
            var storeSum = StoreTimings()
            var storeN = 0
            var last: (t: Double, pb: CVPixelBuffer)?

            func ask(_ q: TestQuestion) async throws {
                guard let last else { return }
                session.setLatest(t: last.t, image: RGBImage(pixelBuffer: last.pb))
                let r = try await session.ask(q.question)
                var rec = TestRecord(
                    id: q.id, run: run, bench: q.bench, task: q.task, question: q.question, reference: q.answer,
                    answer: r.text, timings: r.timings, gridFps: gridFps, stored: session.storedCount,
                    frameTimes: r.frameTimes, now: r.now, storeMean: Self.mean(storeSum, storeN), storeFrames: storeN,
                    memoryMB: 0, peakMemoryMB: 0, thermal: DeviceStats.thermalState, device: DeviceStats.model)
                if multipleChoice, let opts = q.options {
                    let lettered = opts.enumerated().map { "\(Character(UnicodeScalar(65 + $0.offset)!)). \($0.element)" }
                    let m = try await session.ask(q.question, options: lettered)
                    rec.mcAnswer = m.text
                    rec.mcProbs = m.probs
                    rec.mcSeconds = m.seconds
                    if let gt = q.gt { rec.mcCorrect = m.text == String(Character(UnicodeScalar(65 + gt)!)) }
                }
                if measureLatentsAtQuestion {
                    let l = try await session.timeLatentsAtQuestion(now: r.now, thumbnailTimes: r.frameTimes)
                    rec.latentsAtQuestionEmbeds = l.embeds
                    rec.latentsAtQuestionThumbnails = l.thumbnails
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
            while qi < qs.count, let sb = out.copyNextSampleBuffer() {
                guard let pb = CMSampleBufferGetImageBuffer(sb) else { continue }
                let t = CMSampleBufferGetPresentationTimeStamp(sb).seconds
                while qi < qs.count && t > qs[qi].time + 1e-6 {
                    try await ask(qs[qi])
                    qi += 1
                }
                if qi == qs.count { break }
                last = (t, pb)
                if session.wantsStoreFrame(at: t) {
                    let start = Date()
                    try await session.addFrame(t: t, image: RGBImage(pixelBuffer: pb))
                    var s = session.lastStoreTimings
                    // conversion from the pixel buffer counts as preparation
                    s.prep = Date().timeIntervalSince(start) - s.embed - s.thumbnail
                    storeSum.prep += s.prep
                    storeSum.embed += s.embed
                    storeSum.thumbnail += s.thumbnail
                    storeN += 1
                }
            }
            reader.cancelReading()
            while qi < qs.count {
                try await ask(qs[qi])
                qi += 1
            }
        }
    }

    static func mean(_ s: StoreTimings, _ n: Int) -> StoreTimings {
        guard n > 0 else { return s }
        var m = StoreTimings()
        m.prep = s.prep / Double(n)
        m.embed = s.embed / Double(n)
        m.thumbnail = s.thumbnail / Double(n)
        return m
    }
}
