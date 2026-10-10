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
    /// reference tokenizer splits on added tokens the same way).
    public func tokenize(_ text: String) -> [Int] {
        var ids: [Int] = []
        var rest = Substring(text)
        while !rest.isEmpty {
            var hit: (Range<Substring.Index>, String)?
            for tok in specialPattern {
                if let r = rest.range(of: tok), hit == nil || r.lowerBound < hit!.0.lowerBound {
                    hit = (r, tok)
                }
            }
            guard let (r, tok) = hit else {
                ids += tokenizer.encode(text: String(rest), addSpecialTokens: false)
                break
            }
            if r.lowerBound > rest.startIndex {
                ids += tokenizer.encode(text: String(rest[..<r.lowerBound]), addSpecialTokens: false)
            }
            ids.append(spec.specialTokens[tok]!)
            rest = rest[r.upperBound...]
        }
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
        let ids = tokenize(spec.render(parts: parts)) + spec.answerOpenIds
        let features = images.isEmpty
            ? MLXArray.zeros([0, model.config.text.hiddenSize])
            : concatenated(images.map(\.features), axis: 0)
        let emb = model.inputEmbeddings(tokenIDs: ids.map { Int32($0) }, imageFeatures: features)
        let caches = model.makeCaches()
        let hidden = model.lm(embeddings: emb, caches: caches)
        var logits = model.lm.logits(hidden[0..., -1, 0...]).asType(.float32).squeezed()

        if numOptions > 0 {
            let letters = MLXArray(spec.letterIds.prefix(numOptions).map { Int32($0) })
            let probs = softmax(logits[letters], axis: -1).asArray(Float.self).map(Double.init)
            let best = probs.indices.max { probs[$0] < probs[$1] }!
            return Answer(
                text: String(UnicodeScalar(65 + best)!), probs: probs, promptTokens: ids.count,
                visualTokens: features.dim(0))
        }

        let eos = Set(spec.eosTokenIds)
        var generated: [Int] = []
        var next = logits.argMax().item(Int.self)
        while !eos.contains(next) && generated.count < maxNewTokens {
            generated.append(next)
            let tail = decode(Array(generated.suffix(6)))
            if let onText { onText(Self.visible(decode(generated))) }
            if tail.contains(PromptText.answerClose) { break }
            let e = model.lm.embedTokens(MLXArray([Int32(next)])).expandedDimensions(axis: 0)
            let h = model.lm(embeddings: e, caches: caches)
            logits = model.lm.logits(h[0..., -1, 0...]).asType(.float32).squeezed()
            next = logits.argMax().item(Int.self)
        }
        let text = Self.visible(decode(generated))
        onText?(text)
        return Answer(text: text, probs: nil, promptTokens: ids.count, visualTokens: features.dim(0))
    }

    static func visible(_ s: String) -> String {
        (s.components(separatedBy: PromptText.answerClose).first ?? s)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
