// es-verify session <models dir> <frames.safetensors> <reference dir>: the whole Swift StreamSession on the
// reference session's decoded frames (tools/dump_frames.py), asking the reference's questions at the same moments
// (the first frame at or after question time - 0.5 s, as tools/dump_reference.py did), compared with its answers.

import EfficientStreamKit
import Foundation
import MLX

func checkSession(models: URL, frames: URL, ref: URL) async throws {
    let meta = try JSONDecoder().decode(Meta.self, from: Data(contentsOf: ref.appendingPathComponent("meta.json")))
    let times = try JSONDecoder().decode(
        [Double].self, from: Data(contentsOf: frames.deletingPathExtension().appendingPathExtension("json")))
    let d = try loadArrays(url: frames)
    let cache = FileManager.default.temporaryDirectory.appendingPathComponent("es-verify-coreml")
    let es = try await EfficientStream(directory: models, cacheDirectory: cache)
    let session = es.session()
    // question moments: "t60_q1" -> 60
    var byMoment: [Double: [Case]] = [:]
    for c in meta.cases {
        let at = Double(c.tag.dropFirst().prefix { $0 != "_" })!
        byMoment[at, default: []].append(c)
    }
    var pending = byMoment.keys.sorted()
    var storeSeconds: [Double] = []
    var sameAnswers = 0, sameFrames = 0, total = 0
    for (i, t) in times.enumerated() {
        guard !pending.isEmpty else { break }
        let img = rgb(d["f\(i)"]!)
        let stores = session.wantsStoreFrame(at: t)
        let t0 = Date()
        try await session.addFrame(t: t, image: img)
        if stores { storeSeconds.append(Date().timeIntervalSince(t0)) }
        if t + 1e-6 >= pending[0] - 0.5 {
            let at = pending.removeFirst()
            for c in byMoment[at]! {
                let r = try await session.ask(c.question, options: c.options)
                let sameA = r.text == c.answer
                let sameF = r.frameTimes.map { ($0 * 2).rounded() } == c.frame_times.map { ($0 * 2).rounded() }
                sameAnswers += sameA ? 1 : 0
                sameFrames += sameF ? 1 : 0
                total += 1
                print("\(c.tag) at \(r.now)s: frames \(sameF ? "same" : "DIFFERENT") | answer \(sameA ? "same" : "DIFFERENT")"
                    + String(format: " | %.2fs (first token %.2fs), %d tokens", r.seconds, r.secondsToFirstToken, r.promptTokens))
                if !sameF { print("   swift frames \(r.frameTimes)\n   ref   frames \(c.frame_times)") }
                if !sameA { print("   swift: \(r.text) \(r.probs ?? [])\n   ref:   \(c.answer) \(c.probs ?? [])") }
            }
        }
    }
    let mean = storeSeconds.isEmpty ? 0 : storeSeconds.reduce(0, +) / Double(storeSeconds.count)
    print(String(format: "stored %d frames, %.3fs per stored frame on average", session.storedCount, mean))
    print("same retrieved frames \(sameFrames)/\(total), same answers \(sameAnswers)/\(total)")
}
