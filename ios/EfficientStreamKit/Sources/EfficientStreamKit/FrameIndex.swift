// The frame index (efficientstream/index.py): MobileCLIP2-S0 embeds every stored frame once, when it enters the
// store, and the question once, when it is asked. Both towers run as Core ML models (float32) exported by
// tools/export_ios.py; preprocessing and tokenization follow open_clip.

import CoreML
import Foundation

public final class FrameIndex: @unchecked Sendable {
    struct Config: Decodable {
        let imageSize: Int
        let contextLength: Int
        let padTokenId: Int
        enum CodingKeys: String, CodingKey {
            case imageSize = "image_size"
            case contextLength = "context_length"
            case padTokenId = "pad_token_id"
        }
    }

    let image: MLModel
    let text: MLModel
    let config: Config
    public let tokenizer: CLIPTokenizer
    let roundFp16: Bool

    /// `directory` holds ImageEncoder.mlpackage, TextEncoder.mlpackage, vocab.json, merges.txt, config.json.
    /// Compiled models are cached in `cacheDirectory`.
    public init(directory: URL, cacheDirectory: URL, roundFp16: Bool = true) throws {
        config = try JSONDecoder().decode(
            Config.self, from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        let mc = MLModelConfiguration()
        mc.computeUnits = .cpuAndGPU  // float32 model
        image = try MLModel(
            contentsOf: Self.compiled(directory.appendingPathComponent("ImageEncoder.mlpackage"), cacheDirectory),
            configuration: mc)
        text = try MLModel(
            contentsOf: Self.compiled(directory.appendingPathComponent("TextEncoder.mlpackage"), cacheDirectory),
            configuration: mc)
        tokenizer = try CLIPTokenizer(
            vocab: directory.appendingPathComponent("vocab.json"),
            merges: directory.appendingPathComponent("merges.txt"))
        self.roundFp16 = roundFp16
    }

    static func compiled(_ package: URL, _ cache: URL) throws -> URL {
        let target = cache.appendingPathComponent(
            package.deletingPathExtension().lastPathComponent + ".mlmodelc")
        let fm = FileManager.default
        let srcDate = (try? fm.attributesOfItem(atPath: package.path)[.modificationDate] as? Date) ?? .distantPast
        if let d = try? fm.attributesOfItem(atPath: target.path)[.modificationDate] as? Date, d >= srcDate {
            return target
        }
        let tmp = try MLModel.compileModel(at: package)
        try fm.createDirectory(at: cache, withIntermediateDirectories: true)
        try? fm.removeItem(at: target)
        try fm.moveItem(at: tmp, to: target)
        return target
    }

    /// open_clip's transform: shortest side -> image size (bilinear, antialiased), centre crop, [0, 1] (mean 0, std 1).
    public func preprocess(_ im: RGBImage) -> MLMultiArray {
        let s = config.imageSize
        let short = min(im.width, im.height), long = max(im.width, im.height)
        let newLong = Int(Double(s) * Double(long) / Double(short))  // torchvision: int(size * long / short)
        let (w, h) = im.width <= im.height ? (s, newLong) : (newLong, s)
        let resized = im.resized(width: w, height: h, filter: .bilinear)
        let top = Int((Double(h - s) / 2.0).rounded(.toNearestOrEven))
        let left = Int((Double(w - s) / 2.0).rounded(.toNearestOrEven))
        let crop = resized.cropped(x: left, y: top, width: s, height: s)
        let arr = try! MLMultiArray(shape: [1, 3, NSNumber(value: s), NSNumber(value: s)], dataType: .float32)
        let ptr = arr.dataPointer.bindMemory(to: Float.self, capacity: 3 * s * s)
        for y in 0 ..< s {
            for x in 0 ..< s {
                let p = (y * s + x) * 3
                for c in 0 ..< 3 { ptr[c * s * s + y * s + x] = Float(crop.pixels[p + c]) / 255.0 }
            }
        }
        return arr
    }

    func finish(_ out: MLMultiArray) -> [Float] {
        let n = out.count
        let ptr = out.dataPointer.bindMemory(to: Float.self, capacity: n)
        var e = (0 ..< n).map { ptr[$0] }
        let norm = e.reduce(Float(0)) { $0 + $1 * $1 }.squareRoot()
        e = e.map { $0 / max(norm, 1e-12) }
        return roundFp16 ? e.map { Float(Float16($0)) } : e
    }

    public func embed(image im: RGBImage) throws -> [Float] {
        let out = try image.prediction(from: MLDictionaryFeatureProvider(dictionary: ["image": preprocess(im)]))
        return finish(out.featureValue(for: "embedding")!.multiArrayValue!)
    }

    public func embed(text question: String) throws -> [Float] {
        let ids = tokenizer.encode(question, contextLength: config.contextLength)
        let arr = try MLMultiArray(shape: [1, NSNumber(value: ids.count)], dataType: .int32)
        for (i, v) in ids.enumerated() { arr[i] = NSNumber(value: v) }
        let out = try text.prediction(from: MLDictionaryFeatureProvider(dictionary: ["ids": arr]))
        return finish(out.featureValue(for: "embedding")!.multiArrayValue!)
    }
}

/// open_clip's SimpleTokenizer (CLIP byte-level BPE with "</w>" word ends): lower-cased, whitespace-collapsed,
/// [<start>] + tokens + [<end>], truncated / zero-padded to the context length.
public final class CLIPTokenizer: @unchecked Sendable {
    let encoder: [String: Int]
    let ranks: [String: Int]
    let byteEncoder: [UInt8: Character]
    let sot: Int
    let eot: Int
    let pattern: NSRegularExpression
    var cache: [String: [Int]] = [:]

    public init(vocab: URL, merges: URL) throws {
        encoder = try JSONDecoder().decode([String: Int].self, from: Data(contentsOf: vocab))
        let lines = try String(contentsOf: merges, encoding: .utf8).split(separator: "\n", omittingEmptySubsequences: false)
        // SimpleTokenizer: merges[1 : 49152 - 256 - 2 + 1]
        let used = lines.dropFirst().prefix(49152 - 256 - 2)
        var r: [String: Int] = [:]
        for (i, l) in used.enumerated() where !l.isEmpty { r[String(l)] = i }
        ranks = r
        byteEncoder = CLIPTokenizer.bytesToUnicode()
        sot = encoder["<|startoftext|>"]!
        eot = encoder["<|endoftext|>"]!
        pattern = try NSRegularExpression(
            pattern: "<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+",
            options: [.caseInsensitive])
    }

    static func bytesToUnicode() -> [UInt8: Character] {
        var bs: [Int] = Array(33 ... 126) + Array(161 ... 172) + Array(174 ... 255)
        var cs = bs
        var n = 0
        for b in 0 ..< 256 where !bs.contains(b) {
            bs.append(b)
            cs.append(256 + n)
            n += 1
        }
        var m: [UInt8: Character] = [:]
        for (b, c) in zip(bs, cs) { m[UInt8(b)] = Character(UnicodeScalar(c)!) }
        return m
    }

    func bpe(_ token: String) -> [Int] {
        if let c = cache[token] { return c }
        var word = token.map { String($0) }
        word[word.count - 1] += "</w>"
        while word.count > 1 {
            var best: (Int, Int)?
            for i in 0 ..< word.count - 1 {
                if let rk = ranks[word[i] + " " + word[i + 1]], best == nil || rk < best!.0 { best = (rk, i) }
            }
            guard let best else { break }
            let pair = (word[best.1], word[best.1 + 1])
            var merged: [String] = []
            var i = 0
            while i < word.count {
                if i < word.count - 1 && word[i] == pair.0 && word[i + 1] == pair.1 {
                    merged.append(pair.0 + pair.1)
                    i += 2
                } else {
                    merged.append(word[i])
                    i += 1
                }
            }
            word = merged
        }
        let ids = word.compactMap { encoder[$0] }
        cache[token] = ids
        return ids
    }

    public func encode(_ text: String, contextLength: Int = 77) -> [Int] {
        let cleaned = text.components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }
            .joined(separator: " ").lowercased()
        var ids: [Int] = [sot]
        let ns = cleaned as NSString
        for m in pattern.matches(in: cleaned, range: NSRange(location: 0, length: ns.length)) {
            let tok = ns.substring(with: m.range)
            let mapped = String(Array(tok.utf8).map { byteEncoder[$0]! })
            ids += bpe(mapped)
        }
        ids.append(eot)
        if ids.count > contextLength {
            ids = Array(ids.prefix(contextLength))
            ids[contextLength - 1] = eot
        }
        return ids + [Int](repeating: 0, count: contextLength - ids.count)
    }
}
