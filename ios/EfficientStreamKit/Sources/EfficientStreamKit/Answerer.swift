// MiniCPM-V-4.6 as the answering model (efficientstream/model.py): one prefill over the mixed-scale view with the
// assistant turn forced to "<answer>"; multiple choice reads the option letter from the next-token logits,
// open-ended answers decode greedily until "</answer>".

import Foundation
import MLX
import Tokenizers

public struct Answer: Sendable {
    public var text: String
    /// Multiple choice: softmax over the option letters.
    public var probs: [Double]?
    public var promptTokens: Int
    public var visualTokens: Int
    /// Prompt text -> token ids.
    public var tokenizeSeconds: Double = 0
    /// Token embeddings + image features -> logits of the first answer token (evaluated).
    public var prefillSeconds: Double = 0
    /// After the first answer token: the remaining tokens (open-ended only).
    public var decodeSeconds: Double = 0
    /// Answer tokens generated (open-ended; the first one comes from the prefill).
    public var generatedTokens: Int = 0
}

/// An image ready for the prompt: its placeholder text and its features ([tokens, hidden], in placeholder order).
public struct EncodedImage: @unchecked Sendable {
    public let placeholder: String
    public let features: MLXArray
}

public final class Answerer: @unchecked Sendable {
    public let model: MiniCPMV46
    public let spec: PromptSpec
    let tokenizer: Tokenizer
    let specialPattern: [String]

    public init(directory: URL) async throws {
        model = try MiniCPMV46.load(directory: directory)
        spec = try PromptSpec.load(directory.appendingPathComponent("prompt.json"))
        tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        specialPattern = spec.specialTokens.keys.sorted { $0.count > $1.count }
    }

    // MARK: images

    /// Vision features of one image processed with its own resolution and slice budget (the reference processes
    /// each image separately). Views of the same grid run as one batch.
    public func encode(_ image: RGBImage, scaleResolution: Int, maxSlices: Int) -> EncodedImage {
        let processed = MiniCPMImageProcessor.process(
            image, scaleResolution: scaleResolution, maxSlices: maxSlices)
        let tokens = processed.tokens(divisor: spec.imageTokenDivisor)
        let grid = processed.sliceGrid
        let placeholder = spec.placeholder(
            overviewTokens: tokens[0], sliceTokens: tokens.count > 1 ? tokens[1] : 0, rows: grid?.rows ?? 0,
            cols: grid?.cols ?? 0)
        return EncodedImage(placeholder: placeholder, features: features(of: processed.views))
    }

    /// Features of views in order, [sum of tokens, hidden].
    public func features(of views: [ImageView]) -> MLXArray {
        var order: [String: [Int]] = [:]
        var keys: [String] = []
        for (i, v) in views.enumerated() {
            let k = "\(v.gridH)x\(v.gridW)"
            if order[k] == nil { keys.append(k) }
            order[k, default: []].append(i)
        }
        var feats = [MLXArray?](repeating: nil, count: views.count)
        for k in keys {
            let ix = order[k]!
            let group = ix.map { views[$0] }
            let out = model.imageFeatures(
                MiniCPMImageProcessor.pixels(group), gridH: group[0].gridH, gridW: group[0].gridW)
            for (j, i) in ix.enumerated() { feats[i] = out[j] }
        }
        let all = concatenated(feats.map { $0! }, axis: 0)
        eval(all)
        return all
    }

    // MARK: text

    /// Token ids of a prompt string: special tokens by id, the text between them by the tokenizer (the
    /// reference tokenizer splits on added tokens the same way). One pass over the UTF-8 bytes, matching special
    /// tokens where they can start (they all begin with "<").
    public func tokenize(_ text: String) -> [Int] {
        let bytes = Array(text.utf8)
        let specials: [([UInt8], Int)] = specialPattern.map { (Array($0.utf8), spec.specialTokens[$0]!) }
        var ids: [Int] = []
        var segmentStart = 0
        var i = 0
        func flush(_ end: Int) {
            if end > segmentStart {
                let piece = String(decoding: bytes[segmentStart ..< end], as: UTF8.self)
                ids += tokenizer.encode(text: piece, addSpecialTokens: false)
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
        tokenizer.decode(tokens: ids, skipSpecialTokens: true)
    }

    /// Removes the model's special-token strings from user text (a typed "<|image_pad|>" would break the prompt).
    public func sanitize(_ text: String) -> String {
        var t = text
        for tok in specialPattern { t = t.replacingOccurrences(of: tok, with: "") }
        return t
    }

    // MARK: answering

    /// `parts` = the user turn; `images` = the encoded images in the order of the `.image` parts.
    /// `onText` receives the open-ended answer as it grows.
    public func answer(
        parts: [PromptPart], images: [EncodedImage], numOptions: Int, maxNewTokens: Int = 128,
        onText: ((String) -> Void)? = nil
    ) -> Answer {
        let tStart = Date()
        let ids = tokenize(spec.render(parts: parts)) + spec.answerOpenIds
        let tTok = Date()
        let features = images.isEmpty
            ? MLXArray.zeros([0, model.config.text.hiddenSize])
            : concatenated(images.map(\.features), axis: 0)
        let emb = model.inputEmbeddings(tokenIDs: ids.map { Int32($0) }, imageFeatures: features)
        let caches = model.makeCaches()
        let hidden = model.lm(embeddings: emb, caches: caches)
        let logits = model.lm.logits(hidden[0..., -1, 0...]).asType(.float32).squeezed()
        var result = Answer(
            text: "", probs: nil, promptTokens: ids.count, visualTokens: features.dim(0),
            tokenizeSeconds: tTok.timeIntervalSince(tStart))

        if numOptions > 0 {
            let letters = MLXArray(spec.letterIds.prefix(numOptions).map { Int32($0) })
            let probs = softmax(logits[letters], axis: -1).asArray(Float.self).map(Double.init)
            result.prefillSeconds = Date().timeIntervalSince(tTok)
            let best = probs.indices.max { probs[$0] < probs[$1] }!
            result.text = String(UnicodeScalar(65 + best)!)
            result.probs = probs
            return result
        }

        // Greedy decoding. The next step is queued on the GPU (asyncEval) before the current token is read back,
        // so reading a token and decoding its text overlap with the next forward pass.
        let eos = Set(spec.eosTokenIds)
        var generated: [Int] = []
        func step(_ token: MLXArray) -> MLXArray {
            let h = model.lm(embeddings: model.lm.embedTokens(token.reshaped([1])).expandedDimensions(axis: 0),
                             caches: caches)
            return model.lm.logits(h[0..., -1, 0...]).squeezed().argMax()
        }
        var current = logits.argMax()
        eval(current)
        let tFirst = Date()
        result.prefillSeconds = tFirst.timeIntervalSince(tTok)
        while true {
            let next = current.item(Int.self)
            if eos.contains(next) { break }
            generated.append(next)
            if generated.count >= maxNewTokens { break }
            let upcoming = step(current)
            asyncEval(upcoming)
            if let onText { onText(Self.visible(decode(generated))) }
            if decode(Array(generated.suffix(6))).contains(PromptText.answerClose) { break }
            current = upcoming
        }
        result.decodeSeconds = Date().timeIntervalSince(tFirst)
        result.generatedTokens = generated.count
        result.text = Self.visible(decode(generated))
        onText?(result.text)
        return result
    }

    /// One prefill over `embeddings` [1, L, hidden] and the last-position logits (for timing).
    public func benchPrefill(_ embeddings: MLXArray) {
        let caches = model.makeCaches()
        let h = model.lm(embeddings: embeddings, caches: caches)
        eval(model.lm.logits(h[0..., -1, 0...]))
    }

    static func visible(_ s: String) -> String {
        (s.components(separatedBy: PromptText.answerClose).first ?? s)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
