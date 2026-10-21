import AVFoundation
import EfficientStreamKit
import Foundation
import UIKit

/// The streaming Qwen3.5-0.8B on the phone, from the command line (the models in Documents/models/qwen):
/// - `-esqwen pad640,resized100k` runs the streaming test set (Documents/bench, as `-esbench`) once per frame size;
///   answers and timings go to Documents/bench/qwen_results.jsonl, progress prints as ESQWEN lines.
/// - `-esqwenverify` compares the model with mlx-swift-lm's Qwen35 on a few frames (Documents/models/qwen-ref: the
///   checkpoint's config.json and model.safetensors); prints ESQWENVERIFY lines.
@Observable
@MainActor
final class QwenBenchModel {
    struct Line: Identifiable {
        let id = UUID()
        let text: String
    }

    private(set) var status = "Starting…"
    private(set) var done = 0
    private(set) var total = 0
    private(set) var lines: [Line] = []

    static var runs: [String]? {
        let args = ProcessInfo.processInfo.arguments
        guard let i = args.firstIndex(of: "-esqwen"), i + 1 < args.count else { return nil }
        return args[i + 1].split(separator: ",").map(String.init)
    }

    static var verify: Bool { ProcessInfo.processInfo.arguments.contains("-esqwenverify") }

    /// `-esbenches OVO-Bench,StreamingBench`: only these benchmarks' questions.
    static var benches: Set<String>? {
        let args = ProcessInfo.processInfo.arguments
        guard let i = args.firstIndex(of: "-esbenches"), i + 1 < args.count else { return nil }
        return Set(args[i + 1].split(separator: ",").map(String.init))
    }

    /// `-esqwenstream <video file>,<seconds>,<slidingWindow|growing>[;...]` with `-esprompt <text>`: the proactive
    /// mode on a test-set video, one update per second, timed (Documents/bench/qwen_stream.jsonl, ESQWENSTREAM lines).
    static var streams: [(file: String, seconds: Int, context: QwenStreamer.Context)]? {
        let args = ProcessInfo.processInfo.arguments
        guard let i = args.firstIndex(of: "-esqwenstream"), i + 1 < args.count else { return nil }
        return args[i + 1].split(separator: ";").compactMap { spec in
            let p = spec.split(separator: ",").map(String.init)
            guard p.count == 3, let n = Int(p[1]), let c = QwenStreamer.Context(rawValue: p[2]) else { return nil }
            return (p[0], n, c)
        }
    }

    static var streamPrompt: String {
        let args = ProcessInfo.processInfo.arguments
        guard let i = args.firstIndex(of: "-esprompt"), i + 1 < args.count else {
            return "Tell me when the person picks something up."
        }
        return args[i + 1]
    }

    /// `-esqwenprofile`: where a question's time goes inside the language model (ESQWENPROFILE lines).
    static var profile: Bool { ProcessInfo.processInfo.arguments.contains("-esqwenprofile") }

    static var requested: Bool { runs != nil || verify || streams != nil || profile || QwenStreamBench.manifest != nil }

    static var directory: URL { AppModel.documents.appendingPathComponent("models/qwen", isDirectory: true) }

    /// Frame sizes by name: `pad` (the decoded pixels, as the runner), `padN` (scaled down to at most N pixels
    /// first; `pad640` = 640 x 360), `resizedN` (smart_resize to at most N pixels; `resized100k` = the training
    /// windows' 100,352).
    static func size(_ name: String) -> QwenFrameSize? {
        switch name {
        case "pad": return .padded(maxPixels: nil)
        case "pad640": return .padded(maxPixels: 640 * 360)
        case "resized100k": return .resized(minPixels: 3136, maxPixels: 100_352)
        default:
            if name.hasPrefix("pad"), let n = Int(name.dropFirst(3)) { return .padded(maxPixels: n) }
            if name.hasPrefix("resized"), let n = Int(name.dropFirst(7)) { return .resized(minPixels: 3136, maxPixels: n) }
            return nil
        }
    }

    func run(qwen: StreamQwen) async {
        StreamQwen.limitGPUCache(megabytes: 512)
        UIApplication.shared.isIdleTimerDisabled = true
        defer { UIApplication.shared.isIdleTimerDisabled = false }
        if Self.verify { await verify(qwen) }
        if Self.profile { await profile(qwen) }
        if QwenStreamBench.manifest != nil {
            status = "StreamBench (\(QwenStreamBench.context.rawValue))"
            await QwenStreamBench.run(qwen: qwen) { self.say($0) }
        }
        for st in Self.streams ?? [] { await stream(qwen, file: st.file, seconds: st.seconds, context: st.context) }
        for name in Self.runs ?? [] {
            guard let size = Self.size(name) else {
                say("ESQWEN unknown frame size \(name)")
                continue
            }
            status = "Run \(name)"
            do {
                let runner = try QwenTestSetRunner(
                    qwen: qwen, manifest: BenchModel.directory.appendingPathComponent("manifest.json"),
                    videos: BenchModel.directory.appendingPathComponent("videos"),
                    output: BenchModel.directory.appendingPathComponent("qwen_results.jsonl"))
                try await runner.run(name, size: size, limit: BenchModel.limit, benches: Self.benches) { done, total, r in
                    let t = r.timings
                    let line = String(
                        format: "ESQWEN [%@ %d/%d] #%d %@ %d frames %@ | prefill %d tok %.2fs ttft %.2fs total %.2fs "
                            + "%d tok %.1f tok/s | frame prep %.3fs vision %.3fs | mc %@ %.2fs | mem %.0fMB %@ | %@",
                        name, done, total, r.id, r.bench, t.frames, r.framePixels, t.promptTokens, t.prefill,
                        t.timeToFirstToken, t.total, t.generatedTokens, t.decodeTokensPerSecond, t.framePrep,
                        t.frameVision, r.mcAnswer ?? "-", t.mcTotal, r.memoryMB, r.thermal,
                        String(r.answer.prefix(70)))
                    print(line)
                    Task { @MainActor in
                        self.done = done
                        self.total = total
                        self.lines.insert(Line(text: line), at: 0)
                        if self.lines.count > 50 { self.lines.removeLast() }
                    }
                }
            } catch {
                say("ESQWEN FAILED \(error)")
            }
        }
        status = "Finished"
        say("ESQWEN FINISHED")
    }

    /// Two prompts: a 640 x 360 video as decoded (padded), and a 1280 x 720 one scaled to 640 x 360 and to the
    /// training windows' size.
    private func verify(_ qwen: StreamQwen) async {
        status = "Verifying against MLXVLM Qwen35"
        let ref = AppModel.documents.appendingPathComponent("models/qwen-ref", isDirectory: true)
        let cases: [(file: String, times: [Double], question: String, size: QwenFrameSize)] = [
            ("ad51ba17d2.mp4", [16, 17, 18, 19], "What does the poster on the wall say?", .padded(maxPixels: nil)),
            ("c60b7aa354.mp4", [48, 49, 50, 51], "Who just scored?", .padded(maxPixels: 640 * 360)),
            ("c60b7aa354.mp4", [48, 49, 50, 51], "Who just scored?", .resized(minPixels: 3136, maxPixels: 100_352)),
        ]
        for c in cases {
            say("ESQWENVERIFY \(c.file) at \(c.times) \(c.size): \(c.question)")
            do {
                let frames = try await Self.frames(
                    BenchModel.directory.appendingPathComponent("videos").appendingPathComponent(c.file), at: c.times)
                try await QwenVerify.run(qwen: qwen, reference: ref, frames: frames, question: c.question,
                                         size: c.size) { self.say("ESQWENVERIFY " + $0) }
            } catch {
                say("ESQWENVERIFY FAILED \(error)")
            }
            StreamQwen.clearGPUCache()
        }
    }

    /// The proactive mode on the first `seconds` of a video: frames at 4 fps as the runner samples them (the first
    /// frame of each quarter second), one update per second.
    private func stream(_ qwen: StreamQwen, file: String, seconds: Int, context: QwenStreamer.Context) async {
        status = "Proactive: \(file) \(context.rawValue)"
        let run = "\(file)-\(seconds)s-\(context.rawValue)"
        say("ESQWENSTREAM \(run): \(Self.streamPrompt)")
        let out = BenchModel.directory.appendingPathComponent("qwen_stream.jsonl")
        if !FileManager.default.fileExists(atPath: out.path) { FileManager.default.createFile(atPath: out.path, contents: nil) }
        do {
            let handle = try FileHandle(forWritingTo: out)
            defer { try? handle.close() }
            try handle.seekToEnd()
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
            let streamer = QwenStreamer(qwen: qwen, prompt: Self.streamPrompt, context: context)
            let asset = AVURLAsset(url: BenchModel.directory.appendingPathComponent("videos").appendingPathComponent(file))
            guard let track = try await asset.loadTracks(withMediaType: .video).first else { return }
            let reader = try AVAssetReader(asset: asset)
            let output = AVAssetReaderTrackOutput(
                track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
            output.alwaysCopiesSampleData = false
            reader.add(output)
            reader.startReading()
            var second = 0
            var frames: [RGBImage] = []
            var times: [Double] = []
            var lastQuarter = -1
            func flush() throws {
                guard !frames.isEmpty else { return }
                let u = streamer.update(second: second, frames: frames, times: times)
                let line = String(
                    format: "ESQWENSTREAM %@ %3ds | prep %.3fs vision %.3fs | prefill %d tok %.3fs | answer %d tok %.3fs "
                        + "| total %.3fs | context %d%@ | %@",
                    run, u.second, u.prep, u.vision, u.prefillTokens, u.prefill, u.decodeTokens, u.decode, u.total,
                    u.contextTokens, u.recomputed ? " (window recomputed)" : "", String(u.answer.prefix(80)))
                print(line)
                lines.insert(Line(text: line), at: 0)
                var rec = try JSONSerialization.jsonObject(with: encoder.encode(u)) as! [String: Any]
                rec["run"] = run
                rec["prompt"] = Self.streamPrompt
                rec["thermal"] = DeviceStats.thermalState
                rec["memoryMB"] = DeviceStats.memoryMB().current
                try handle.write(contentsOf: JSONSerialization.data(withJSONObject: rec, options: [.sortedKeys])
                    + Data("\n".utf8))
                frames = []
                times = []
            }
            while let sb = output.copyNextSampleBuffer() {
                guard let pb = CMSampleBufferGetImageBuffer(sb) else { continue }
                let t = CMSampleBufferGetPresentationTimeStamp(sb).seconds
                let quarter = Int((t * 4 + 1e-7).rounded(.down))
                guard quarter > lastQuarter else { continue }
                lastQuarter = quarter
                if Int(t.rounded(.down)) > second {
                    try flush()
                    second = Int(t.rounded(.down))
                    if second >= seconds { break }
                }
                frames.append(RGBImage(pixelBuffer: pb))
                times.append(t)
            }
            if second < seconds { try flush() }
            reader.cancelReading()
        } catch {
            say("ESQWENSTREAM FAILED \(error)")
        }
        StreamQwen.clearGPUCache()
    }

    /// The language model's parts on a 16-frame question (the kitchen video at 4:23), for both frame sizes,
    /// 3 times each (the first warms up).
    private func profile(_ qwen: StreamQwen) async {
        status = "Profiling"
        let url = BenchModel.directory.appendingPathComponent("videos").appendingPathComponent("379fc8fe52.mp4")
        guard let raw = try? await Self.frames(url, at: (248 ... 263).map(Double.init)) else { return }
        for (name, size) in [("pad640", Self.size("pad640")!), ("resized100k", Self.size("resized100k")!)] {
            let frames = raw.map { qwen.encode($0.image, time: $0.time, size: size) }
            for run in 0 ..< 3 {
                let r = qwen.profile(frames: frames, question: "Where did I pick the plate from?")
                func fmt(_ d: [String: Double]) -> String {
                    d.sorted { $0.value > $1.value }.map { String(format: "%@ %.4f", $0.key, $0.value) }
                        .joined(separator: " | ")
                }
                say("ESQWENPROFILE \(name) run \(run) prompt \(r.promptTokens) tokens | prefill: " + fmt(r.prefill))
                say("ESQWENPROFILE \(name) run \(run) per decoded token: " + fmt(r.decode))
            }
            let v = frames.map(\.visionSeconds).sorted()
            say(String(format: "ESQWENPROFILE %@ vision tower per frame: median %.4f s (%@)", name, v[v.count / 2],
                       frames.first.map { "\($0.width)x\($0.height), \($0.tokens) tokens" } ?? ""))
        }
    }

    /// The first frame at or after each time.
    static func frames(_ url: URL, at times: [Double]) async throws -> [(time: Double, image: RGBImage)] {
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else { return [] }
        let reader = try AVAssetReader(asset: asset)
        let out = AVAssetReaderTrackOutput(
            track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        reader.add(out)
        reader.startReading()
        var result: [(Double, RGBImage)] = []
        var i = 0
        while i < times.count, let sb = out.copyNextSampleBuffer() {
            guard let pb = CMSampleBufferGetImageBuffer(sb) else { continue }
            let t = CMSampleBufferGetPresentationTimeStamp(sb).seconds
            if t + 1e-6 >= times[i] {
                result.append((t, RGBImage(pixelBuffer: pb)))
                i += 1
            }
        }
        reader.cancelReading()
        return result
    }

    private func say(_ s: String) {
        print(s)
        lines.insert(Line(text: s), at: 0)
    }
}
