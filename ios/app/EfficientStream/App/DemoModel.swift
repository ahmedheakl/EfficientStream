import Foundation
import UIKit

/// A scripted demo for screen recordings. Launched with `-esdemo <script>` (a JSON file under Documents), it plays
/// each segment's video in the normal stream screen, fast-forwards between questions (`rate`; 1 plays everything in
/// real time), slows to real time a few seconds before each one, types the question into the question bar and sends
/// it. The answers are the model's, live. Progress prints as ESDEMO lines with the wall-clock time, to find the
/// moments in a recording; `pos` lines give the playback position four times a second.
///
/// ```json
/// {"rate": 4, "segments": [{"video": "bench/videos/abc.mp4", "start": 0, "rate": 1,
///                           "questions": [{"at": 120, "text": "What did he just do?"}]}]}
/// ```
@MainActor
enum DemoModel {
    struct Script: Decodable {
        var rate: Float?
        /// Seconds between stored frames (default 2, the evaluated setting).
        var interval: Double?
        var segments: [Segment]
    }

    struct Segment: Decodable {
        var video: String
        var start: Double?
        /// Playback speed between questions (default: the script's).
        var rate: Float?
        var questions: [Question]
        /// Seconds to keep showing the last answer.
        var hold: Double?
    }

    struct Question: Decodable {
        var at: Double
        var text: String
    }

    static var script: URL? {
        let args = ProcessInfo.processInfo.arguments
        guard let i = args.firstIndex(of: "-esdemo"), i + 1 < args.count else { return nil }
        return AppModel.documents.appendingPathComponent(args[i + 1])
    }

    static func run(_ c: StreamController, script url: URL) async {
        UIApplication.shared.isIdleTimerDisabled = true
        guard let data = try? Data(contentsOf: url), let script = try? JSONDecoder().decode(Script.self, from: data)
        else {
            log("FAILED reading \(url.lastPathComponent)")
            return
        }
        c.indexInterval = script.interval ?? 2
        let positions = Task {
            while !Task.isCancelled {
                log(String(format: "pos %.3f %.0f %d", c.video.position, c.video.rate, c.video.isPlaying ? 1 : 0))
                try? await Task.sleep(for: .milliseconds(250))
            }
        }
        defer { positions.cancel() }
        try? await Task.sleep(for: .seconds(2))
        for (i, seg) in script.segments.enumerated() {
            let fast = seg.rate ?? script.rate ?? 4
            c.clearHistory()
            c.video.setRate(fast)
            log("segment \(i) \(seg.video)")
            await c.openVideo(AppModel.documents.appendingPathComponent(seg.video), start: seg.start ?? 0)
            for (j, q) in seg.questions.enumerated() {
                let typing = Double(q.text.count) * 0.055
                await until { c.video.position >= q.at - typing - 3 }
                c.video.setRate(1)
                await until { c.video.position >= q.at - typing }
                log("type \(q.text)")
                var typed = ""
                for ch in q.text {
                    typed.append(ch)
                    c.demoTyping = typed
                    let pause = ch == " " ? Double.random(in: 0.06 ... 0.12) : Double.random(in: 0.03 ... 0.065)
                    try? await Task.sleep(for: .seconds(pause))
                }
                try? await Task.sleep(for: .seconds(0.35))
                log("send at \(String(format: "%.2f", c.video.position))")
                let asked = c.history.count
                c.demoSend()
                c.demoTyping = nil
                // the question bar sends on its next update: wait for the new question, then for its answer
                guard await until(timeout: 2, { c.history.count > asked }) else {
                    log("not asked (no frame yet)")
                    continue
                }
                await until { c.history.first.map { !$0.isAnswering } ?? false }
                if let r = c.history.first?.result {
                    log(String(format: "answered ttft %.2f total %.2f %.1f tok/s | %@", r.timings.timeToFirstToken,
                               r.timings.total, r.timings.decodeTokensPerSecond, r.text))
                    log("recalled " + r.closestTimes.map { String(format: "%.1f", $0) }.joined(separator: " ")
                        + String(format: " | now %.1f, %d stored", r.now, r.timings.candidates))
                }
                if j + 1 < seg.questions.count { c.video.setRate(fast) }
            }
            try? await Task.sleep(for: .seconds(seg.hold ?? 3))
            c.video.pause()
        }
        log("DONE")
    }

    private static func until(_ condition: () -> Bool) async {
        while !condition() { try? await Task.sleep(for: .milliseconds(20)) }
    }

    /// False if `condition` did not hold within `timeout` seconds.
    private static func until(timeout: Double, _ condition: () -> Bool) async -> Bool {
        let end = Date().addingTimeInterval(timeout)
        while !condition() {
            if Date() > end { return false }
            try? await Task.sleep(for: .milliseconds(20))
        }
        return true
    }

    private static func log(_ s: String) {
        print(String(format: "ESDEMO %.3f ", Date().timeIntervalSince1970) + s)
    }
}
