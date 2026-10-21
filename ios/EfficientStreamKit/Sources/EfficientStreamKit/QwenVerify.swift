// Checks the Kit's streaming Qwen3.5-0.8B against mlx-swift-lm's MLXVLM Qwen35 (an independent MLX port of the
// same transformers model, loading the original checkpoint) on the phone: the same prompt ids and patch pixels go
// into both, and the last-position logits and the greedy answers are compared. (There is no PyTorch reference: the
// models are not run on the Mac.)

import Foundation
import MLX
import MLXLMCommon
import MLXVLM

public enum QwenVerify {
    /// `reference`: a folder with the checkpoint's config.json and model.safetensors (Hugging Face layout).
    public static func run(
        qwen: StreamQwen, reference: URL, frames: [(time: Double, image: RGBImage)], question: String,
        size: QwenFrameSize, answerTokens: Int = 24, log: (String) -> Void
    ) async throws {
        // both models stay in memory: keep MLX's buffer cache small (the app's limit is about 6 GB)
        MLX.GPU.set(cacheLimit: 256 * 1024 * 1024)
        defer { MLX.GPU.clearCache() }
        // the same frames for both: sized, patches, and (ours) features
        var ours: [QwenFrame] = []
        var patches: [MLXArray] = []
        var grids: [THW] = []
        for f in frames {
            let im = qwen.sized(f.image, size)
            let (x, gh, gw) = qwen.patches([im])
            patches.append(x)
            grids.append(THW(1, gh, gw))
            ours.append(qwen.encode(f.image, time: f.time, size: size))
        }
        let prompt = qwen.askPrompt(frames: ours, question: question, options: nil)
        let (ids, positions) = qwen.expand(qwen.tokenize(prompt), frames: ours)
        log("prompt: \(ids.count) tokens, \(ours.map(\.tokens).reduce(0, +)) vision tokens, frames "
            + ours.map { "\($0.width)x\($0.height)" }.joined(separator: " "))

        // ours: the first token's logits and the greedy tokens (as ids), then `ask` (the answer as the app gets it)
        let m = qwen.model
        let caches = m.makeCaches()
        var logits = m.logits(m.prefill(
            m.inputEmbeddings(tokenIDs: ids, features: concatenated(ours.map(\.features), axis: 0)),
            positions: positions, caches: caches)).asType(.float32).reshaped(-1)
        eval(logits)
        let ourLogits = logits
        var ourTokens: [Int] = []
        var pos = positions.next
        for _ in 0 ..< answerTokens {
            let t = logits.argMax().item(Int.self)
            ourTokens.append(t)
            if qwen.spec.eosTokenIds.contains(t) { break }
            logits = m.logits(m.step(MLXArray(Int32(t)), position: pos, caches: caches)).reshaped(-1)
            pos += 1
        }
        let ourAnswer = qwen.ask(frames: ours, question: question, options: nil, maxNewTokens: answerTokens)

        // reference
        MLX.GPU.clearCache()
        let config = try JSONDecoder().decode(
            Qwen35Configuration.self, from: Data(contentsOf: reference.appendingPathComponent("config.json")))
        let ref = Qwen35(config)
        try await loadWeights(modelDirectory: reference, model: ref)
        let input = LMInput(
            text: .init(tokens: MLXArray(ids).expandedDimensions(axis: 0)),
            image: .init(pixels: concatenated(patches, axis: 0), frames: grids))
        let cache = try ref.newCache(parameters: nil)
        guard case .logits(let out) = try ref.prepare(
            input, cache: cache, state: nil, prefill: PrefillParameters(chunking: .unchunked))
        else { throw NSError(domain: "QwenVerify", code: 1) }
        var refLogits = out.logits[0..., -1, 0...].asType(.float32).reshaped(-1)
        eval(refLogits)
        let refFirst = refLogits
        var refTokens: [Int] = []
        var state = out.state
        for _ in 0 ..< answerTokens {
            let t = refLogits.argMax().item(Int.self)
            refTokens.append(t)
            if qwen.spec.eosTokenIds.contains(t) { break }
            let o = ref(.init(tokens: MLXArray([Int32(t)]).expandedDimensions(axis: 0)), cache: cache, state: state)
            state = o.state
            refLogits = o.logits[0..., -1, 0...].asType(.float32).reshaped(-1)
        }

        // compare
        let a = ourLogits, b = refFirst
        let cosine = ((a * b).sum() / (sqrt((a * a).sum()) * sqrt((b * b).sum()))).item(Float.self)
        let maxDiff = abs(a - b).max().item(Float.self)
        let topA = argSort(-a)[0 ..< 10].asArray(Int32.self)
        let topB = argSort(-b)[0 ..< 10].asArray(Int32.self)
        let pa = softmax(a), pb = softmax(b)
        let kl = (pb * (MLX.log(pb + 1e-12) - MLX.log(pa + 1e-12))).sum().item(Float.self)
        var same = 0
        while same < min(ourTokens.count, refTokens.count) && ourTokens[same] == refTokens[same] { same += 1 }
        log(String(format: "first-token logits: cosine %.6f, max |diff| %.3f, KL(ref||ours) %.5f", cosine, maxDiff, kl))
        log("top-10 ours \(topA)")
        log("top-10 ref  \(topB)")
        log("greedy tokens identical for \(same) of \(max(ourTokens.count, refTokens.count))")
        log("ours: \(StreamQwen.visible(qwen.decode(ourTokens)))")
        log("ref : \(StreamQwen.visible(qwen.decode(refTokens)))")
        log("ask(): \(ourAnswer.text)")
    }
}
