// es-verify index <models dir> <index.safetensors>: the frame index against tools/dump_reference.py --index.

import CoreML
import EfficientStreamKit
import Foundation
import MLX

struct IndexMeta: Decodable {
    let times: [Double]
    let questions: [String]
}

func dot(_ a: [Float], _ b: [Float]) -> Float { zip(a, b).reduce(0) { $0 + $1.0 * $1.1 } }

func checkIndex(models: URL, file: URL) throws {
    let d = try loadArrays(url: file)
    let meta = try JSONDecoder().decode(
        IndexMeta.self, from: Data(contentsOf: file.deletingPathExtension().appendingPathExtension("json")))
    let cache = FileManager.default.temporaryDirectory.appendingPathComponent("es-verify-coreml")
    let index = try FrameIndex(directory: models.appendingPathComponent("mobileclip"), cacheDirectory: cache)
    let cfg = try StreamConfig.load(models.appendingPathComponent("efficientstream_config.json"))

    // questions: token ids and text embeddings
    var idsSame = 0
    var textCos: [Float] = []
    var qSwift: [[Float]] = [], qRef: [[Float]] = []
    for (j, q) in meta.questions.enumerated() {
        let ref = d["qids\(j)"]!.asArray(Int32.self).map(Int.init)
        let ids = index.tokenizer.encode(q, contextLength: 77)
        if ids == ref { idsSame += 1 } else { print("  ids differ for \"\(q)\":\n   swift \(ids.prefix(20))\n   ref   \(ref.prefix(20))") }
        let e = try index.embed(text: q)
        let r = d["qemb\(j)"]!.asArray(Float.self)
        textCos.append(dot(e, r))
        qSwift.append(e)
        qRef.append(r)
    }
    print("questions: token ids identical \(idsSame)/\(meta.questions.count); text embedding cosine min \(textCos.min()!)")

    // frames: preprocessing, embeddings of the reference's stored frames, and the full Swift store path
    var preDiff: Float = 0, storedCos: [Float] = [], pathCos: [Float] = [], resizeDiff = 0
    var refEmbs: [[Float]] = [], swiftEmbs: [[Float]] = []
    for i in 0 ..< meta.times.count {
        let stored = rgb(d["stored\(i)"]!)
        let pre = index.preprocess(stored)
        let p = pre.dataPointer.bindMemory(to: Float.self, capacity: pre.count)
        let rp = d["pre\(i)"]!.reshaped(-1).asArray(Float.self)
        for k in 0 ..< pre.count { preDiff = max(preDiff, abs(p[k] - rp[k])) }
        let r = d["emb\(i)"]!.asArray(Float.self)
        storedCos.append(dot(try index.embed(image: stored), r))
        refEmbs.append(r)
        // raw frame -> store size (Pillow-exact) -> JPEG (ImageIO) -> embedding
        let raw = rgb(d["raw\(i)"]!)
        let resized = raw.resizedForStore(maxPixels: cfg.gridMaxPixels)
        let rr = d["resized\(i)"]!.reshaped(-1).asArray(UInt8.self)
        for (x, y) in zip(resized.pixels, rr) { resizeDiff = max(resizeDiff, abs(Int(x) - Int(y))) }
        let e = try index.embed(image: resized.jpegRoundTrip(quality: cfg.storeJpegQuality))
        pathCos.append(dot(e, r))
        swiftEmbs.append(e)
    }
    print("frames: preprocessing max |diff| \(preDiff); embedding of reference stored frames cosine min \(storedCos.min()!)")
    print("store path (raw -> resize -> ImageIO JPEG -> embed): resize max pixel diff \(resizeDiff); embedding cosine min \(pathCos.min()!) mean \(pathCos.reduce(0, +) / Float(pathCos.count))")

    // retrieval: top-16 with all-Swift embeddings vs all-reference embeddings
    var overlaps: [Int] = []
    for j in 0 ..< meta.questions.count {
        let a = topKInTimeOrder(swiftEmbs, query: qSwift[j], k: cfg.retrieveK)
        let b = topKInTimeOrder(refEmbs, query: qRef[j], k: cfg.retrieveK)
        overlaps.append(Set(a).intersection(b).count)
    }
    print("retrieval top-\(cfg.retrieveK) of \(meta.times.count) frames, same frames per question: \(overlaps)")
}
