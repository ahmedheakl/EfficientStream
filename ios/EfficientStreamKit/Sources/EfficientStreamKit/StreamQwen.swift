// The streaming Qwen3.5-0.8B's question answering (proactive/stream_qwen.py `ask`): a question on the most recent
// frames (one per second, oldest to newest, each with its time), answered from those frames only. Frames are
// encoded by the vision tower when they arrive, so a question costs only the prompt's prefill and the answer.

import Foundation
import MLX
import Tokenizers

/// prompt.json written by tools/export_qwen.py: special tokens and the runner's system prompts.
public struct QwenPromptSpec: Codable, Sendable {
    public var specialTokens: [String: Int]
    public var eosTokenIds: [Int]
    public var letterIds: [Int]
    public var stateTokens: [String: Int]
    public var imageMean: [Float]
    public var imageStd: [Float]
    public var streamSystem: String
    public var qaSystem: String
    public var qaDirectSystem: String
    public var postPrompt: String
    public var minPixels: Int
    public var maxPixels: Int
    /// The caption memory (PHCM), run-4 weights on: nil for earlier exports.
    public var captionSystem: String?
    public var captionInstruction: String?
    public var memoryPreamble: String?
    public var memoryPostamble: String?
    public var memoryMaxTokens: Int?

    enum CodingKeys: String, CodingKey {
        case specialTokens = "special_tokens"
        case eosTokenIds = "eos_token_ids"
        case letterIds = "letter_ids"
        case stateTokens = "state_tokens"
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case streamSystem = "STREAM_SYSTEM"
        case qaSystem = "QA_SYSTEM"
        case qaDirectSystem = "QA_DIRECT_SYSTEM"
        case postPrompt = "POST_PROMPT"
        case minPixels = "MIN_PIXELS"
        case maxPixels = "MAX_PIXELS"
        case captionSystem = "CAPTION_SYSTEM"
        case captionInstruction = "CAPTION_INSTRUCTION"
        case memoryPreamble = "MEMORY_PREAMBLE"
        case memoryPostamble = "MEMORY_POSTAMBLE"
        case memoryMaxTokens = "MEMORY_MAX_TOKENS"
    }
}

/// How a frame is sized for the vision tower (both give a multiple of 32 pixels per side).
public enum QwenFrameSize: Sendable, CustomStringConvertible {
    /// As the runner's `ask`: the decoded pixels, padded with black (bottom, right) to a multiple of 32. With
    /// `maxPixels`, a larger frame is first scaled down (bicubic, aspect kept) to fit.
    case padded(maxPixels: Int?)
    /// Qwen's smart_resize to a multiple of 32 within [min, max] pixels (the streaming windows' frames).
    case resized(minPixels: Int, maxPixels: Int)

    public var description: String {
        switch self {
        case .padded(let m): return m.map { "padded<=\($0)px" } ?? "padded"
        case .resized(let a, let b): return "resized\(a)-\(b)px"
        }
    }
}

/// A frame through the vision tower: its features [rows * cols, hidden] (merged tokens, row-major).
public struct QwenFrame: @unchecked Sendable {
    public let time: Double
    public let features: MLXArray
    public let rows: Int
    public let cols: Int
    /// Pixels given to the vision tower.
    public let width: Int
    public let height: Int
    /// Resize / pad and patches, and the vision tower (evaluated).
    public var prepSeconds: Double
    public var visionSeconds: Double
    public var tokens: Int { rows * cols }
}

public struct QwenAnswer: Sendable {
    public var text: String
    /// Multiple choice: softmax over the option letters.
    public var probs: [Double]?
    public var promptTokens: Int
    public var visualTokens: Int
    public var tokenizeSeconds: Double = 0
    /// Embeddings + prefill -> the first answer token's logits (evaluated).
    public var prefillSeconds: Double = 0
    /// The remaining tokens (open-ended).
    public var decodeSeconds: Double = 0
    public var generatedTokens: Int = 0
    /// The decoded answer before any tag was removed (questions on a stream).
    public var raw = ""
}

public final class StreamQwen: @unchecked Sendable {
    public let model: Qwen35VL
    public let spec: QwenPromptSpec
    let tokenizer: Tokenizer
    private let specials: [([UInt8], Int)]
    /// uint8 -> normalized value ((x / 255 - mean) / std, float32), per channel.
    private let normTable: MLXArray

    public init(directory: URL) async throws {
        model = try Qwen35VL.load(directory: directory)
        spec = try JSONDecoder().decode(
            QwenPromptSpec.self, from: Data(contentsOf: directory.appendingPathComponent("prompt.json")))
        tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        specials = spec.specialTokens.sorted { $0.key.count > $1.key.count }.map { (Array($0.key.utf8), $0.value) }
        var table = [Float]()
        for c in 0 ..< 3 {
            for v in 0 ..< 256 {
                table.append((Float(v) * (1 / 255) - spec.imageMean[c]) / spec.imageStd[c])
            }
        }
        normTable = MLXArray(table, [3, 256])
    }

    /// Releases MLX's cached GPU buffers (between independent runs).
    public static func clearGPUCache() { MLX.GPU.clearCache() }

    /// Caps MLX's cache of freed GPU buffers (by default it keeps every size a prompt used; prompts of varying
    /// length then grow the app's memory to the per-app limit).
    public static func limitGPUCache(megabytes: Int) { MLX.GPU.set(cacheLimit: megabytes * 1024 * 1024) }

    public static func modelsPresent(in directory: URL) -> Bool {
        ["config.json", "model.safetensors", "prompt.json", "tokenizer.json"].allSatisfy {
            FileManager.default.fileExists(atPath: directory.appendingPathComponent($0).path)
        }
    }

    // MARK: frames

    /// The frame as the vision tower takes it (a multiple of 32 pixels per side).
    public func sized(_ image: RGBImage, _ size: QwenFrameSize) -> RGBImage {
        let f = model.config.tokenSide
        switch size {
        case .padded(let maxPixels):
            var im = image
            if let maxPixels, im.width * im.height > maxPixels {
                let s = (Double(maxPixels) / Double(im.width * im.height)).squareRoot()
                im = im.resized(width: max(1, Int((Double(im.width) * s).rounded())),
                                height: max(1, Int((Double(im.height) * s).rounded())))
            }
            return im.paddedBottomRight(toMultipleOf: f)
        case .resized(let minPixels, let maxPixels):
            let (h, w) = smartResize(height: image.height, width: image.width, factor: f, minPixels: minPixels,
                                     maxPixels: maxPixels)
            return image.width == w && image.height == h ? image : image.resized(width: w, height: h)
        }
    }

    /// Patch vectors [rows * cols * 4, 3 * 2 * 16 * 16] (bfloat16) of one frame (repeated over the temporal patch, as
    /// the image processor does) or two frames (one temporal patch of a video), in the processor's order.
    public func patches(_ frames: [RGBImage]) -> (patches: MLXArray, gridH: Int, gridW: Int) {
        let (w, h) = (frames[0].width, frames[0].height)
        let p = model.config.vision.patchSize
        let m = model.config.vision.spatialMergeSize
        let t = model.config.vision.temporalPatchSize
        var raw = frames.map { MLXArray($0.pixels, [h, w, 3]).asType(.int32) }
        while raw.count < t { raw.append(raw[raw.count - 1]) }
        // per-channel lookup: index = channel * 256 + value
        let x = stacked(raw, axis: 0) + MLXArray([0, 256, 512] as [Int32])
        let v = normTable.reshaped(-1)[x]  // [T, H, W, 3] float32
        let (gh, gw) = (h / p, w / p)
        let out = v.reshaped(t, gh / m, m, p, gw / m, m, p, 3)
            .transposed(1, 4, 2, 5, 7, 0, 3, 6)
            .reshaped(gh * gw, 3 * t * p * p)
        return (out.asType(.bfloat16), gh, gw)
    }

    /// One frame through the vision tower (evaluated), timed.
    public func encode(_ image: RGBImage, time: Double, size: QwenFrameSize) -> QwenFrame {
        let t0 = Date()
        let im = sized(image, size)
        let (x, gh, gw) = patches([im])
        eval(x)
        let t1 = Date()
        let feats = model.vision(x.expandedDimensions(axis: 0), gridH: gh, gridW: gw)[0]
        eval(feats)
        let m = model.config.vision.spatialMergeSize
        return QwenFrame(time: time, features: feats, rows: gh / m, cols: gw / m, width: im.width, height: im.height,
                         prepSeconds: t1.timeIntervalSince(t0), visionSeconds: Date().timeIntervalSince(t1))
    }

    // MARK: text

    /// Token ids of a prompt string: special tokens by id, the text between them by the tokenizer.
    public func tokenize(_ text: String) -> [Int] {
        let bytes = Array(text.utf8)
        var ids: [Int] = []
        var segmentStart = 0
        var i = 0
        func flush(_ end: Int) {
            if end > segmentStart {
                ids += tokenizer.encode(
                    text: String(decoding: bytes[segmentStart ..< end], as: UTF8.self), addSpecialTokens: false)
            }
        }
        while i < bytes.count {
            if bytes[i] == UInt8(ascii: "<"),
                let (tok, id) = specials.first(where: { t in
                    i + t.0.count <= bytes.count && bytes[i ..< i + t.0.count].elementsEqual(t.0)
                })
            {
                flush(i)
                ids.append(id)
                i += tok.count
                segmentStart = i
            } else {
                i += 1
            }
        }
        flush(bytes.count)
        return ids
    }

    public func decode(_ ids: [Int]) -> String {
        tokenizer.decode(tokens: ids, skipSpecialTokens: false)
    }

    /// The runner's `ask` prompt (Qwen3-VL chat template; one image placeholder per frame). `memory`: the caption
    /// memory's text (`CaptionMemory.text`, PHCM), or nil for the frames alone.
    public func askPrompt(frames: [QwenFrame], question: String, options: [String]?, memory: String? = nil) -> String {
        var user = (memory ?? "<LONG_TERM_VIDEO_MEMORY>None for this ablation.</LONG_TERM_VIDEO_MEMORY>")
            + "<RECENT_VISUAL_CONTEXT>\nThe following frames are ordered from oldest to newest. They retain their "
            + "decoded source pixels; only minimal border padding may be added to align them to the model patch grid."
        for f in frames {
            user += String(format: "<%.3fs>", f.time) + "<|vision_start|><|image_pad|><|vision_end|>"
        }
        let q = options.map {
            (["Question:\n\(question)", "Options:"] + $0 + [spec.postPrompt, "Best option:("]).joined(separator: "\n")
        } ?? question
        user += "</RECENT_VISUAL_CONTEXT>\n" + q
        let system = options == nil ? spec.qaDirectSystem : spec.qaSystem
        return "<|im_start|>system\n\(system)<|im_end|>\n<|im_start|>user\n\(user)<|im_end|>\n<|im_start|>assistant\n"
    }

    /// Token ids with each image placeholder expanded to its frame's tokens, and their M-RoPE positions.
    public func expand(_ ids: [Int], frames: [QwenFrame], start: Int = 0) -> ([Int32], MRopePositions) {
        let pad = model.config.imageTokenID
        var out: [Int32] = []
        var pos = MRopePositions(start: start)
        var fi = 0
        for id in ids {
            if id == pad {
                let f = frames[fi]
                fi += 1
                out += [Int32](repeating: Int32(pad), count: f.tokens)
                pos.vision(rows: f.rows, cols: f.cols)
            } else {
                out.append(Int32(id))
                pos.text(1)
            }
        }
        precondition(fi == frames.count, "\(fi) image placeholders for \(frames.count) frames")
        return (out, pos)
    }

    // MARK: answering

    /// A question on `frames` (oldest to newest). `options` = lines like "A. a cup" (the answer is the letter with
    /// the highest logit); open-ended answers decode greedily to <|im_end|> (at most `maxNewTokens`).
    public func ask(
        frames: [QwenFrame], question: String, options: [String]?, memory: String? = nil, maxNewTokens: Int = 64,
        onText: ((String) -> Void)? = nil
    ) -> QwenAnswer {
        let t0 = Date()
        let (ids, positions) = expand(
            tokenize(askPrompt(frames: frames, question: question, options: options, memory: memory)), frames: frames)
        let t1 = Date()
        let feats = frames.isEmpty ? nil : concatenated(frames.map(\.features), axis: 0)
        let caches = model.makeCaches()
        let hidden = model.prefill(model.inputEmbeddings(tokenIDs: ids, features: feats), positions: positions,
                                   caches: caches)
        let logits = model.logits(hidden).asType(.float32).reshaped(-1)
        var result = QwenAnswer(
            text: "", promptTokens: ids.count, visualTokens: feats?.dim(0) ?? 0,
            tokenizeSeconds: t1.timeIntervalSince(t0))

        if let options {
            let letters = MLXArray(spec.letterIds.prefix(options.count).map { Int32($0) })
            let probs = softmax(logits[letters], axis: -1).asArray(Float.self).map(Double.init)
            result.prefillSeconds = Date().timeIntervalSince(t1)
            let best = probs.indices.max { probs[$0] < probs[$1] }!
            result.text = String(UnicodeScalar(65 + best)!)
            result.probs = probs
            return result
        }

        // greedy decoding, the next step queued (asyncEval) before the current token is read
        let eos = Set(spec.eosTokenIds)
        var generated: [Int] = []
        var position = positions.next
        var current = logits.argMax()
        eval(current)
        let tFirst = Date()
        result.prefillSeconds = tFirst.timeIntervalSince(t1)
        while true {
            let next = current.item(Int.self)
            if eos.contains(next) { break }
            generated.append(next)
            if generated.count >= maxNewTokens { break }
            let upcoming = model.logits(model.step(current, position: position, caches: caches)).reshaped(-1).argMax()
            position += 1
            asyncEval(upcoming)
            if let onText { onText(Self.visible(decode(generated))) }
            current = upcoming
        }
        result.decodeSeconds = Date().timeIntervalSince(tFirst)
        result.generatedTokens = generated.count
        result.text = Self.visible(decode(generated))
        onText?(result.text)
        return result
    }

    /// Where a question's time goes inside the language model: the `ask` prompt on `frames` prefilled, then
    /// `decodeTokens` greedy tokens (not stopped at the end of the answer), each part evaluated and timed on its own.
    public func profile(frames: [QwenFrame], question: String, decodeTokens: Int = 32)
        -> (prefill: [String: Double], decode: [String: Double], promptTokens: Int)
    {
        let (ids, positions) = expand(tokenize(askPrompt(frames: frames, question: question, options: nil)),
                                      frames: frames)
        let feats = concatenated(frames.map(\.features), axis: 0)
        let caches = model.makeCaches()
        let pre = LayerProfiler()
        LayerProfiler.active = pre
        let t0 = Date()
        let emb = model.inputEmbeddings(tokenIDs: ids, features: feats)
        eval(emb)
        let tEmb = Date().timeIntervalSince(t0)
        var current = model.logits(model.prefill(emb, positions: positions, caches: caches, chunk: 1 << 20))
            .reshaped(-1).argMax()
        eval(current)
        var prefill = pre.seconds
        prefill["embeddings (tokens + vision features)"] = tEmb
        let dec = LayerProfiler()
        LayerProfiler.active = dec
        var position = positions.next
        for _ in 0 ..< decodeTokens {
            current = model.logits(model.step(current, position: position, caches: caches)).reshaped(-1).argMax()
            eval(current)
            position += 1
        }
        LayerProfiler.active = nil
        return (prefill, dec.seconds.mapValues { $0 / Double(decodeTokens) }, ids.count)
    }

    static func visible(_ s: String) -> String {
        var t = s
        for end in ["<|im_end|>", "<|endoftext|>"] { t = t.components(separatedBy: end).first ?? t }
        return t.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

/// Qwen2-VL's smart_resize: the closest size with both sides multiples of `factor` and the pixel count within
/// [minPixels, maxPixels] (Python's round, half to even).
public func smartResize(height: Int, width: Int, factor: Int, minPixels: Int, maxPixels: Int) -> (Int, Int) {
    let (h, w, f) = (Double(height), Double(width), Double(factor))
    var hBar = (h / f).rounded(.toNearestOrEven) * f
    var wBar = (w / f).rounded(.toNearestOrEven) * f
    if hBar * wBar > Double(maxPixels) {
        let beta = (h * w / Double(maxPixels)).squareRoot()
        hBar = max(f, (h / beta / f).rounded(.down) * f)
        wBar = max(f, (w / beta / f).rounded(.down) * f)
    } else if hBar * wBar < Double(minPixels) {
        let beta = (Double(minPixels) / (h * w)).squareRoot()
        hBar = (h * beta / f).rounded(.up) * f
        wBar = (w * beta / f).rounded(.up) * f
    }
    return (Int(hBar), Int(wBar))
}

extension RGBImage {
    /// Black padding on the bottom and right to multiples of `m` (PIL ImageOps.expand(border=(0, 0, r, b), fill=0)).
    public func paddedBottomRight(toMultipleOf m: Int) -> RGBImage {
        let (w2, h2) = ((width + m - 1) / m * m, (height + m - 1) / m * m)
        if w2 == width && h2 == height { return self }
        var out = [UInt8](repeating: 0, count: w2 * h2 * 3)
        pixels.withUnsafeBufferPointer { src in
            out.withUnsafeMutableBufferPointer { dst in
                for y in 0 ..< height {
                    let s = y * width * 3
                    let d = y * w2 * 3
                    for i in 0 ..< width * 3 { dst[d + i] = src[s + i] }
                }
            }
        }
        return RGBImage(width: w2, height: h2, pixels: out)
    }
}
