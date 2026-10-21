import AVFoundation
import EfficientStreamKit
import Foundation

/// StreamBench on the caption stream, as the app runs it with no watch prompt: `-esqwensb <manifest>` (a JSON file
/// under Documents/sb: `[{"video": "x.mp4", "questions": [{"id": 0, "time": 28, "question": "…", "answer": "…",
/// "class": "SF"}]}]`, videos in Documents/sb/videos) with `-escontext compress|compress4|evict8` (default compress: 2 frames kept) and
/// `-esfps`. Each video goes through the caption stream second by second as fast as the phone runs it (frames at
/// `fps`, the first of each 1/fps slot); a question at time t is asked after the seconds before t, on the stream's
/// caches with the captions of the seconds without frames as text, as the app's question bar does. One JSON line per
/// second (`"kind": "second"`) and per question (`"question"`) goes to Documents/sb/results_<context>.jsonl; videos
/// already there are skipped (a run can be resumed). Progress prints as ESQWENSB lines.
@MainActor
enum QwenStreamBench {
    struct Video: Decodable {
        var video: String
        var questions: [Question]
    }

    struct Question: Decodable {
        var id: Int
        var time: Double
        var question: String
        var answer: String
        var `class`: String
    }

    static var manifest: String? {
        let args = ProcessInfo.processInfo.arguments
        guard let i = args.firstIndex(of: "-esqwensb"), i + 1 < args.count else { return nil }
        return args[i + 1]
    }

    static var context: QwenStreamer.Context {
        let args = ProcessInfo.processInfo.arguments
        guard let i = args.firstIndex(of: "-escontext"), i + 1 < args.count,
              let c = QwenStreamer.Context(rawValue: args[i + 1]) else { return .compress(2) }
        return c
    }

    static var directory: URL { AppModel.documents.appendingPathComponent("sb", isDirectory: true) }

    static func run(qwen: StreamQwen, say: @escaping (String) -> Void) async {
        guard let name = manifest,
              let data = try? Data(contentsOf: directory.appendingPathComponent(name)),
              let videos = try? JSONDecoder().decode([Video].self, from: data)
        else {
            say("ESQWENSB FAILED reading the manifest")
            return
        }
        let ctx = context
        let fps = QwenStreamController.launchFPS
        let out = directory.appendingPathComponent("results_\(ctx.rawValue).jsonl")
        if !FileManager.default.fileExists(atPath: out.path) { FileManager.default.createFile(atPath: out.path, contents: nil) }
        let done = Set((try? String(contentsOf: out, encoding: .utf8))?.split(separator: "\n").compactMap { line in
            (try? JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any])
                .flatMap { $0["kind"] as? String == "video" ? $0["video"] as? String : nil }
        } ?? [])
        guard let handle = try? FileHandle(forWritingTo: out) else { return }
        defer { try? handle.close() }
        _ = try? handle.seekToEnd()
        func write(_ rec: [String: Any]) {
            if let d = try? JSONSerialization.data(withJSONObject: rec, options: [.sortedKeys]) {
                handle.write(d + Data("\n".utf8))
            }
        }
        let spec = qwen.spec
        let size = QwenFrameSize.resized(minPixels: spec.minPixels, maxPixels: spec.maxPixels)
        say("ESQWENSB \(videos.count) videos, \(videos.reduce(0) { $0 + $1.questions.count }) questions, "
            + "\(ctx.rawValue), \(fps) fps, \(done.count) done before")
        for (vi, v) in videos.enumerated() where !done.contains(v.video) {
            let t0 = Date()
            let stream = QwenStreamer(qwen: qwen, prompt: nil, context: ctx)
            let memory = CaptionMemory()
            var questions = v.questions.sorted { $0.time < $1.time }
            func ask(upTo t: Double) {
                while let q = questions.first, q.time <= t {
                    questions.removeFirst()
                    guard let snap = stream.snapshot() else {
                        write(["kind": "question", "video": v.video, "id": q.id, "error": "no stream yet"])
                        continue
                    }
                    let older = memory.all.filter { $0.end <= Double(snap.from) }
                    let text = older.isEmpty ? nil : CaptionMemory.text(older, preamble: spec.memoryPreamble ?? "",
                                                                         postamble: spec.memoryPostamble ?? "")
                    let a0 = Date()
                    let (a, _, _) = stream.ask(q.question, on: snap, memory: text)
                    let total = Date().timeIntervalSince(a0)
                    let ttft = total - a.decodeSeconds
                    write(["kind": "question", "video": v.video, "id": q.id, "class": q.class, "time": q.time,
                           "question": q.question, "reference": q.answer, "answer": a.text, "raw": a.raw,
                           "ttft": ttft, "total": total, "prefill": a.tokenizeSeconds + a.prefillSeconds,
                           "decode": a.decodeSeconds, "promptTokens": a.promptTokens, "tokens": a.generatedTokens,
                           "contextTokens": snap.contextTokens, "from": snap.from, "through": snap.through,
                           "frames": snap.frames, "captions": older.count, "thermal": DeviceStats.thermalState])
                    say(String(format: "ESQWENSB [%d/%d] #%d %@ t=%.0f first words %.2fs total %.2fs ctx %d tok "
                                   + "%d frames %d captions | %@", vi + 1, videos.count, q.id, q.class, q.time, ttft,
                               total, snap.contextTokens, snap.frames, older.count, String(a.text.prefix(80))))
                }
            }
            do {
                let asset = AVURLAsset(url: directory.appendingPathComponent("videos").appendingPathComponent(v.video))
                guard let track = try await asset.loadTracks(withMediaType: .video).first else {
                    say("ESQWENSB no video track in \(v.video)")
                    continue
                }
                let reader = try AVAssetReader(asset: asset)
                let output = AVAssetReaderTrackOutput(
                    track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
                output.alwaysCopiesSampleData = false
                reader.add(output)
                reader.startReading()
                let last = v.questions.map(\.time).max() ?? 0
                var second = 0
                var frames: [RGBImage] = []
                var times: [Double] = []
                var lastSlot = -1
                var stats: [Double] = []
                func flush() {
                    guard !frames.isEmpty else { return }
                    ask(upTo: Double(second))
                    let sec = QwenSecond(qwen: qwen, size: size, second: second, frames: frames, times: times)
                    let u = stream.update(sec)
                    memory.add(u.answer, second: Double(second), end: Double(second + 1))
                    stats.append(u.total)
                    var rec: [String: Any] = [
                        "kind": "second", "video": v.video, "second": second, "total": u.total, "vision": u.vision,
                        "prep": u.prep, "prefill": u.prefill, "prefillTokens": u.prefillTokens, "decode": u.decode,
                        "decodeTokens": u.decodeTokens, "compress": u.compress, "contextTokens": u.contextTokens,
                        "answer": u.answer,
                    ]
                    if let k = stream.lastKept, k.second == second - 1 {
                        rec["kept"] = k.blocks.first ?? 0
                        rec["keptBlocks"] = k.blocks
                        rec["shares"] = k.shares
                    }
                    write(rec)
                    frames = []
                    times = []
                }
                while let sb = output.copyNextSampleBuffer() {
                    guard let pb = CMSampleBufferGetImageBuffer(sb) else { continue }
                    let t = CMSampleBufferGetPresentationTimeStamp(sb).seconds
                    let slot = Int((t * Double(fps) + 1e-7).rounded(.down))
                    guard slot > lastSlot else { continue }
                    lastSlot = slot
                    if Int(t.rounded(.down)) > second {
                        flush()
                        second = Int(t.rounded(.down))
                        if Double(second) >= last { break }
                    }
                    frames.append(RGBImage(pixelBuffer: pb))
                    times.append(t)
                }
                if Double(second) < last { flush() }
                reader.cancelReading()
                ask(upTo: .infinity)
                let sorted = stats.sorted()
                write(["kind": "video", "video": v.video, "seconds": stats.count, "wall": Date().timeIntervalSince(t0),
                       "thermal": DeviceStats.thermalState])
                say(String(format: "ESQWENSB video %d/%d %@ done: %d s of video in %.0f s, per second median %.3f s",
                           vi + 1, videos.count, v.video, stats.count, Date().timeIntervalSince(t0),
                           sorted.isEmpty ? 0 : sorted[sorted.count / 2]))
            } catch {
                say("ESQWENSB FAILED \(v.video): \(error)")
            }
            StreamQwen.clearGPUCache()
        }
        say("ESQWENSB FINISHED")
    }
}
