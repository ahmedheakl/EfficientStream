import EfficientStreamKit
import Foundation
import UIKit

/// A scripted demo of the streaming Qwen3.5-0.8B for screen recordings. Launched with
/// `-model qwen -esqwendemo <script>` (a JSON file under Documents): for each segment it opens the video paused at
/// `start`, types the watch prompt (if any) into the watch field and starts watching, then plays in real time for
/// `seconds`, typing each question into the question bar so that it is sent at video time `at`. The updates and
/// answers are the model's own, live. Every step, update and answer prints as an ESQDEMO line with the wall-clock
/// time, and `pos` lines give the playback position four times a second, to find the moments in a recording.
///
/// ```json
/// {"segments": [{"video": "bench/videos/abc.mp4", "start": 0, "seconds": 14,
///                "prompt": "Tell me when the person picks up a plate.",
///                "questions": [{"at": 9, "text": "What is the person holding?"}]}]}
/// ```
@MainActor
enum QwenDemoModel {
    struct Script: Decodable {
        var segments: [Segment]
    }

    struct Segment: Decodable {
        var video: String
        var start: Double?
        var seconds: Double
        var prompt: String?
        var questions: [Question]?
        /// Playback speed before questions (no watch prompt): slows to real time 3 s before each one.
        var rate: Float?
        /// Seconds to keep showing the end.
        var hold: Double?
        /// Puts the watch prompt in at once instead of typing it (benchmark runs).
        var instant: Bool?
        /// A label for the logs (benchmark item id).
        var id: String?
    }

    struct Question: Decodable {
        var at: Double
        var text: String
    }

    static var script: URL? {
        let args = ProcessInfo.processInfo.arguments
        guard let i = args.firstIndex(of: "-esqwendemo"), i + 1 < args.count else { return nil }
        return AppModel.documents.appendingPathComponent(args[i + 1])
    }

    static func run(_ c: QwenStreamController, script url: URL) async {
        UIApplication.shared.isIdleTimerDisabled = true
        guard let data = try? Data(contentsOf: url), let script = try? JSONDecoder().decode(Script.self, from: data)
        else {
            log("FAILED reading \(url.lastPathComponent)")
            return
        }
        c.onUpdate = { u in
            log(String(format: "update %d %.3f %d %@", u.second, u.total, u.decodeTokens, u.answer))
            if let r = u.repeated { log("repeat \(u.second) \(r)") }
        }
        c.onCaption = { u in
            log(String(format: "caption %d %.3f %d %@", u.second, u.total, u.decodeTokens,
                       u.answer.replacingOccurrences(of: "\n", with: " ")))
        }
        let positions = Task {
            while !Task.isCancelled {
                log(String(format: "pos %.3f 1 %d", c.video.position, c.video.isPlaying ? 1 : 0))
                try? await Task.sleep(for: .milliseconds(250))
            }
        }
        defer { positions.cancel() }
        // `-esdelay <s>`: wait before starting, e.g. to start a screen recorder after launching
        let args = ProcessInfo.processInfo.arguments
        let delay = args.firstIndex(of: "-esdelay").flatMap { $0 + 1 < args.count ? Double(args[$0 + 1]) : nil } ?? 0
        log("waiting \(delay)")
        try? await Task.sleep(for: .seconds(2 + delay))
        for (i, seg) in script.segments.enumerated() {
            c.watch(nil)
            c.clearHistory()
            // benchmark runs: let a hot phone cool down to "fair" first, and log the state each item ran at
            if seg.instant == true {
                var waited = 0
                while ProcessInfo.processInfo.thermalState.rawValue >= 2 {
                    if waited % 30 == 0 { log("cooling \(ProcessInfo.processInfo.thermalState.rawValue)") }
                    try? await Task.sleep(for: .seconds(1))
                    waited += 1
                }
            }
            log("segment \(i) \(seg.video) id \(seg.id ?? "-") thermal \(ProcessInfo.processInfo.thermalState.rawValue)")
            await c.openVideo(AppModel.documents.appendingPathComponent(seg.video), play: false)
            if let s = seg.start, s > 0 { await c.video.seek(to: s) }
            try? await Task.sleep(for: .seconds(1))
            if let prompt = seg.prompt {
                log("type \(prompt)")
                if seg.instant == true { c.demoWatchTyping = prompt } else { await type(prompt) { c.demoWatchTyping = $0 } }
                try? await Task.sleep(for: .seconds(0.35))
                log("start")
                c.demoStartWatch()
            }
            let fast = seg.prompt == nil ? seg.rate ?? 1 : 1
            c.video.setRate(fast)
            c.video.play()
            log("play")
            for q in (seg.questions ?? []).sorted(by: { $0.at < $1.at }) {
                let typing = Double(q.text.count) * 0.055 + 0.35
                await until { c.video.position >= q.at - typing - 3 }
                c.video.setRate(1)
                await until { c.video.position >= q.at - typing }
                log("ask \(q.text)")
                await type(q.text) { c.demoTyping = $0 }
                try? await Task.sleep(for: .seconds(0.35))
                log(String(format: "send at %.2f", c.video.position))
                let asked = c.history.count
                c.demoSend()
                c.demoTyping = nil
                guard await until(timeout: 2, { c.history.count > asked }), let qa = c.history.first else {
                    log("not asked (no frame yet)")
                    continue
                }
                // the answer arrives while the video keeps playing (and the watch keeps updating)
                Task {
                    await until { !qa.isAnswering }
                    if let r = qa.result {
                        log(String(format: "answered ttft %.2f total %.2f %d tok %d frames | %@",
                                   qa.seconds - r.decodeSeconds, qa.seconds, r.generatedTokens, qa.frames, r.text))
                        log(String(format: "timing memory %d events | prefill %.3f decode %.3f, %d prompt tokens",
                                   qa.memoryEvents, r.tokenizeSeconds + r.prefillSeconds, r.decodeSeconds,
                                   r.promptTokens))
                        if let c = qa.context {
                            log(String(format: "context %d %d %d tokens %d frames %d behind %@ | raw %@", c.from,
                                       c.through, c.tokens, qa.frames, qa.behind, qa.onWatch ? "watch" : "caption",
                                       r.raw))
                        }
                    }
                }
                // fast again until the next question (a memory segment); the watch, if any, plays in real time
                if fast > 1 {
                    await until { !qa.isAnswering }
                    c.video.setRate(fast)
                }
            }
            let end = (seg.start ?? 0) + seg.seconds
            while c.video.position < end { try? await Task.sleep(for: .milliseconds(50)) }
            c.video.pause()
            log("pause")
            try? await Task.sleep(for: .seconds(seg.hold ?? 3))
        }
        log("DONE")
    }

    /// Types `text` a character at a time at a typing pace, reporting what is typed so far.
    private static func type(_ text: String, _ show: (String) -> Void) async {
        var typed = ""
        for ch in text {
            typed.append(ch)
            show(typed)
            let pause = ch == " " ? Double.random(in: 0.06 ... 0.12) : Double.random(in: 0.03 ... 0.065)
            try? await Task.sleep(for: .seconds(pause))
        }
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

    /// Also written to Documents/demo/last_demo.log, for runs launched without a console attached.
    private static let logFile: FileHandle? = {
        let url = AppModel.documents.appendingPathComponent("demo/last_demo.log")
        FileManager.default.createFile(atPath: url.path, contents: nil)
        return try? FileHandle(forWritingTo: url)
    }()

    private static func log(_ s: String) {
        let line = String(format: "ESQDEMO %.3f ", Date().timeIntervalSince1970) + s
        print(line)
        logFile?.write(Data((line + "\n").utf8))
    }
}
