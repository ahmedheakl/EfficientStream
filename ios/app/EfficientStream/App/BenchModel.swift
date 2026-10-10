import EfficientStreamKit
import Foundation
import UIKit

/// The streaming test set on the phone. Launched with `-esbench <runs>` (e.g. `-esbench 0.5,1`: one run at
/// 0.5 fps, one at 1 fps; optional `-eslimit N`), with Documents/bench/manifest.json and Documents/bench/videos/
/// copied from the Mac (es-bench). Answers and timings go to Documents/bench/results.jsonl, one line per question.
@Observable
@MainActor
final class BenchModel {
    struct Line: Identifiable {
        let id = UUID()
        let text: String
    }

    private(set) var status = "Starting…"
    private(set) var done = 0
    private(set) var total = 0
    private(set) var lines: [Line] = []
    private(set) var finished = false

    static var directory: URL { AppModel.documents.appendingPathComponent("bench", isDirectory: true) }

    /// Runs requested on the command line (grid fps values), or nil.
    static var requestedRuns: [Double]? {
        let args = ProcessInfo.processInfo.arguments
        guard let i = args.firstIndex(of: "-esbench"), i + 1 < args.count else { return nil }
        return args[i + 1].split(separator: ",").compactMap { Double($0) }
    }

    static var limit: Int? {
        let args = ProcessInfo.processInfo.arguments
        guard let i = args.firstIndex(of: "-eslimit"), i + 1 < args.count else { return nil }
        return Int(args[i + 1])
    }

    func run(es: EfficientStream, runs: [Double]) async {
        UIApplication.shared.isIdleTimerDisabled = true
        defer { UIApplication.shared.isIdleTimerDisabled = false }
        do {
            let runner = try TestSetRunner(
                es: es, manifest: Self.directory.appendingPathComponent("manifest.json"),
                videos: Self.directory.appendingPathComponent("videos"),
                output: Self.directory.appendingPathComponent("results.jsonl"))
            for fps in runs {
                let name = String(format: "%gfps", fps)
                status = "Run \(name)"
                // question-time latents are measured once, in the evaluated setting
                try await runner.run(name, gridFps: fps, measureLatentsAtQuestion: fps == 0.5, limit: Self.limit) {
                    done, total, r in
                    let t = r.timings
                    let line = String(
                        format: "[%@ %d/%d] #%d %@ ttft %.2fs total %.2fs %d tok %.1f tok/s mem %.0fMB %@ | %@",
                        name, done, total, r.id, r.bench, t.timeToFirstToken, t.total, t.generatedTokens,
                        t.decodeTokensPerSecond, r.memoryMB, r.thermal, String(r.answer.prefix(60)))
                    print("ESBENCH " + line)
                    Task { @MainActor in
                        self.done = done
                        self.total = total
                        self.lines.insert(Line(text: line), at: 0)
                        if self.lines.count > 50 { self.lines.removeLast() }
                    }
                }
            }
            status = "Finished"
            print("ESBENCH FINISHED")
        } catch {
            status = "Failed: \(error)"
            print("ESBENCH FAILED \(error)")
        }
        finished = true
    }
}
