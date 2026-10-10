// es-verify testset <models dir> <bench dir> <run name> <grid fps> [limit]: runs the streaming test set
// (<bench dir>/manifest.json, <bench dir>/videos/) on this Mac; appends to <bench dir>/results-mac.jsonl.

import EfficientStreamKit
import Foundation

func runTestSet(models: URL, bench: URL, run: String, gridFps: Double, limit: Int?) async throws {
    let es = try await EfficientStream(
        directory: models, cacheDirectory: FileManager.default.temporaryDirectory.appendingPathComponent("es-coreml"))
    let runner = try TestSetRunner(
        es: es, manifest: bench.appendingPathComponent("manifest.json"),
        videos: bench.appendingPathComponent("videos"), output: bench.appendingPathComponent("results-mac.jsonl"))
    try await runner.run(run, gridFps: gridFps, measureLatentsAtQuestion: true, limit: limit) { done, total, r in
        let t = r.timings
        print(String(format: "[%d/%d] %@ #%d  ttft %.2fs  total %.2fs  %d tok  %.1f tok/s  | %@",
                     done, total, r.bench, r.id, t.timeToFirstToken, t.total, t.generatedTokens,
                     t.decodeTokensPerSecond, String(r.answer.prefix(80))))
    }
}
