// The streaming Qwen3.5-0.8B (Qwen3_5ForConditionalGeneration) on MLX: Qwen3.5's vision tower (16-pixel patches
// over 2 frames, a learned 48x48 position grid interpolated to the frame, 2D rotary attention, 2x2 merger into the
// language model's space) and the Qwen3.5 text model shared with MiniCPM-V-4.6 (TextModel), with interleaved
// M-RoPE positions for the vision tokens.
//
// Mirrors transformers' `modeling_qwen3_5.py`. Weights are the checkpoint's, renamed and re-laid out by
// `tools/export_qwen.py` (patch embedding as a linear layer over (channel, frame, row, column) patch vectors, the
// processor's order; depthwise conv weights as [channels, kernel, 1]; RMSNorm weights with the +1 offset folded in).

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Configuration

public struct Qwen35Config: Codable, Sendable {
    public struct Vision: Codable, Sendable {
        public var hiddenSize: Int
        public var intermediateSize: Int
        public var depth: Int
        public var numHeads: Int
        public var patchSize: Int
        public var temporalPatchSize: Int
        public var spatialMergeSize: Int
        public var numPositionEmbeddings: Int
        public var outHiddenSize: Int

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case depth
            case numHeads = "num_heads"
            case patchSize = "patch_size"
            case temporalPatchSize = "temporal_patch_size"
            case spatialMergeSize = "spatial_merge_size"
            case numPositionEmbeddings = "num_position_embeddings"
            case outHiddenSize = "out_hidden_size"
        }
    }

    public var vision: Vision
    public var text: MiniCPMVConfig.Text
    public var mropeSection: [Int]
    public var imageTokenID: Int
    public var videoTokenID: Int
    public var visionStartTokenID: Int
    public var visionEndTokenID: Int

    /// Pixels per merged vision token along a side (patch size x spatial merge).
    public var tokenSide: Int { vision.patchSize * vision.spatialMergeSize }

    enum CodingKeys: String, CodingKey {
        case vision = "vision_config"
        case text = "text_config"
        case mropeSection = "mrope_section"
        case imageTokenID = "image_token_id"
        case videoTokenID = "video_token_id"
        case visionStartTokenID = "vision_start_token_id"
        case visionEndTokenID = "vision_end_token_id"
    }
}

// MARK: - Vision tower

/// x [F, P, D] -> rotate-half rotary with cos/sin [P, D] (both halves the same table), in float32 as the reference.
private func visionRotary(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
    let d = x.dim(-1)
    let xf = x.asType(.float32)
    let x1 = xf[.ellipsis, 0 ..< d / 2]
    let x2 = xf[.ellipsis, (d / 2) ..< d]
    return (xf * cos + concatenated([-x2, x1], axis: -1) * sin).asType(x.dtype)
}

final class QwenVisionAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var proj: Linear

    init(_ c: Qwen35Config.Vision) {
        numHeads = c.numHeads
        headDim = c.hiddenSize / c.numHeads
        scale = pow(Float(headDim), -0.5)
        _qkv.wrappedValue = Linear(c.hiddenSize, 3 * c.hiddenSize, bias: true)
        _proj.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: true)
    }

    /// x [F, P, D]: each of the F frames attends within itself (the reference's cu_seqlens per frame).
    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let (f, p) = (x.dim(0), x.dim(1))
        let parts = qkv(x).reshaped(f, p, 3, numHeads, headDim)
        let c = cos.reshaped(1, p, 1, headDim)
        let s = sin.reshaped(1, p, 1, headDim)
        let q = visionRotary(parts[0..., 0..., 0], cos: c, sin: s).transposed(0, 2, 1, 3)
        let k = visionRotary(parts[0..., 0..., 1], cos: c, sin: s).transposed(0, 2, 1, 3)
        let v = parts[0..., 0..., 2].transposed(0, 2, 1, 3)
        let o = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: .none)
        return proj(o.transposed(0, 2, 1, 3).reshaped(f, p, -1))
    }
}

final class QwenVisionMLP: Module {
    @ModuleInfo(key: "linear_fc1") var fc1: Linear
    @ModuleInfo(key: "linear_fc2") var fc2: Linear

    init(_ c: Qwen35Config.Vision) {
        _fc1.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: true)
        _fc2.wrappedValue = Linear(c.intermediateSize, c.hiddenSize, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(geluApproximate(fc1(x)))  // gelu_pytorch_tanh
    }
}

final class QwenVisionBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "attn") var attn: QwenVisionAttention
    @ModuleInfo(key: "mlp") var mlp: QwenVisionMLP

    init(_ c: Qwen35Config.Vision) {
        _norm1.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: 1e-6)
        _norm2.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: 1e-6)
        _attn.wrappedValue = QwenVisionAttention(c)
        _mlp.wrappedValue = QwenVisionMLP(c)
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let h = x + attn(norm1(x), cos: cos, sin: sin)
        return h + mlp(norm2(h))
    }
}

/// LayerNorm per patch, then each 2x2 block of patches (consecutive in the processor's order) concatenated and
/// projected into the language model's space.
final class QwenVisionMerger: Module {
    let unit: Int

    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "linear_fc1") var fc1: Linear
    @ModuleInfo(key: "linear_fc2") var fc2: Linear

    init(_ c: Qwen35Config.Vision) {
        unit = c.spatialMergeSize * c.spatialMergeSize
        let merged = c.hiddenSize * unit
        _norm.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: 1e-6)
        _fc1.wrappedValue = Linear(merged, merged, bias: true)
        _fc2.wrappedValue = Linear(merged, c.outHiddenSize, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (f, p, d) = (x.dim(0), x.dim(1), x.dim(2))
        return fc2(gelu(fc1(norm(x).reshaped(f, p / unit, d * unit))))
    }
}

public final class QwenVisionTower: Module {
    let config: Qwen35Config.Vision

    @ModuleInfo(key: "patch_embed") var patchEmbed: Linear
    @ModuleInfo(key: "pos_embed") var posEmbed: Embedding
    let blocks: [QwenVisionBlock]
    @ModuleInfo(key: "merger") var merger: QwenVisionMerger

    /// Position embeddings and rotary tables per patch grid (rows, columns).
    private var geometry: [Int: (pos: MLXArray, cos: MLXArray, sin: MLXArray)] = [:]
    private let lock = NSLock()

    init(_ c: Qwen35Config.Vision) {
        config = c
        let patchDim = 3 * c.temporalPatchSize * c.patchSize * c.patchSize
        _patchEmbed.wrappedValue = Linear(patchDim, c.hiddenSize, bias: true)
        _posEmbed.wrappedValue = Embedding(embeddingCount: c.numPositionEmbeddings, dimensions: c.hiddenSize)
        blocks = (0 ..< c.depth).map { _ in QwenVisionBlock(c) }
        _merger.wrappedValue = QwenVisionMerger(c)
    }

    /// patches [F, P, 3 * T * 16 * 16] of F frames (or frame pairs) sharing one grid of `gridH` x `gridW` patches
    /// (P = gridH * gridW, in the processor's 2x2-block order) -> [F, P / 4, out hidden].
    public func callAsFunction(_ patches: MLXArray, gridH: Int, gridW: Int) -> MLXArray {
        let g = tables(gridH: gridH, gridW: gridW)
        var x = patchEmbed(patches) + g.pos.asType(patches.dtype)
        for block in blocks {
            x = block(x, cos: g.cos, sin: g.sin)
        }
        return merger(x)
    }

    private func tables(gridH: Int, gridW: Int) -> (pos: MLXArray, cos: MLXArray, sin: MLXArray) {
        let key = gridH * 100_000 + gridW
        if let g = lock.withLock({ geometry[key] }) { return g }
        let (cos, sin) = rotaryTables(gridH: gridH, gridW: gridW)
        let out = (pos: positionEmbeddings(gridH: gridH, gridW: gridW), cos: cos, sin: sin)
        eval(out.pos, out.cos, out.sin)
        lock.withLock { geometry[key] = out }
        return out
    }

    /// The learned 48x48 grid, bilinearly interpolated to gridH x gridW (corners aligned: torch.linspace(0, 47, n)),
    /// in the merge-block order [P, D]. Weighted in the embedding's dtype as the reference.
    private func positionEmbeddings(gridH: Int, gridW: Int) -> MLXArray {
        let side = Int(Double(config.numPositionEmbeddings).squareRoot())
        func axis(_ n: Int) -> (lo: [Int], hi: [Int], frac: [Float]) {
            var lo = [Int](), hi = [Int](), frac = [Float]()
            for i in 0 ..< n {
                let v = n == 1 ? Float(0) : linspaceValue(i, n, Float(side - 1))
                let f = Int(v)
                lo.append(f)
                hi.append(min(f + 1, side - 1))
                frac.append(v - Float(f))
            }
            return (lo, hi, frac)
        }
        let (hl, hh, dh) = axis(gridH)
        let (wl, wh, dw) = axis(gridW)
        var idx = [[Int32]](repeating: [], count: 4)
        var wts = [[Float]](repeating: [], count: 4)
        for r in 0 ..< gridH {
            for c in 0 ..< gridW {
                idx[0].append(Int32(hl[r] * side + wl[c]))
                idx[1].append(Int32(hl[r] * side + wh[c]))
                idx[2].append(Int32(hh[r] * side + wl[c]))
                idx[3].append(Int32(hh[r] * side + wh[c]))
                wts[0].append((1 - dh[r]) * (1 - dw[c]))
                wts[1].append((1 - dh[r]) * dw[c])
                wts[2].append(dh[r] * (1 - dw[c]))
                wts[3].append(dh[r] * dw[c])
            }
        }
        let table = posEmbed.weight
        var sum: MLXArray?
        for i in 0 ..< 4 {
            let term = table[MLXArray(idx[i])] * MLXArray(wts[i]).asType(table.dtype).expandedDimensions(axis: -1)
            sum = sum.map { $0 + term } ?? term
        }
        let m = config.spatialMergeSize
        return sum!.reshaped(gridH / m, m, gridW / m, m, -1).transposed(0, 2, 1, 3, 4).reshaped(gridH * gridW, -1)
    }

    /// Rotary cos/sin [P, head dim]: row frequencies then column frequencies (theta 10000 over head dim / 2),
    /// duplicated for the two halves, per patch in the merge-block order.
    private func rotaryTables(gridH: Int, gridW: Int) -> (MLXArray, MLXArray) {
        let headDim = config.hiddenSize / config.numHeads
        let dim = headDim / 2
        let invFreq = stride(from: 0, to: dim, by: 2).map { 1 / pow(Float(10000), Float($0) / Float(dim)) }
        let m = config.spatialMergeSize
        var rows = [Float](), cols = [Float]()
        for br in 0 ..< gridH / m {
            for bc in 0 ..< gridW / m {
                for ir in 0 ..< m {
                    for ic in 0 ..< m {
                        rows.append(Float(br * m + ir))
                        cols.append(Float(bc * m + ic))
                    }
                }
            }
        }
        let f = MLXArray(invFreq)
        let freqs = concatenated(
            [MLXArray(rows).expandedDimensions(axis: 1) * f, MLXArray(cols).expandedDimensions(axis: 1) * f], axis: 1)
        let emb = concatenated([freqs, freqs], axis: 1)
        return (MLX.cos(emb), MLX.sin(emb))
    }
}

/// torch.linspace(0, end, n)[i] in float32 (the first half counted from the start, the second from the end).
func linspaceValue(_ i: Int, _ n: Int, _ end: Float) -> Float {
    let step = end / Float(n - 1)
    return i < n / 2 ? Float(i) * step : end - Float(n - 1 - i) * step
}

// MARK: - Positions (M-RoPE)

/// The language model's 3D positions (time, row, column) for a prompt, as transformers' get_rope_index: text tokens
/// advance one position each (all three equal); a vision block of h x w merged tokens takes time = p, row = p + r,
/// column = p + c, then the next position is p + max(h, w).
public struct MRopePositions {
    public var t: [Int32] = []
    public var h: [Int32] = []
    public var w: [Int32] = []
    /// The next free position (text after these tokens starts here).
    public var next: Int = 0

    public init(start: Int = 0) { next = start }

    public var count: Int { t.count }

    public mutating func text(_ n: Int) {
        for i in 0 ..< n {
            let p = Int32(next + i)
            t.append(p)
            h.append(p)
            w.append(p)
        }
        next += n
    }

    public mutating func vision(rows: Int, cols: Int) {
        let p = Int32(next)
        for r in 0 ..< rows {
            for c in 0 ..< cols {
                t.append(p)
                h.append(p + Int32(r))
                w.append(p + Int32(c))
            }
        }
        next += max(rows, cols)
    }

    /// cos/sin [count, rotary dims] for the text model: interleaved sections (frequency i takes the time position if
    /// i % 3 == 0, the row if 1, the column if 2, within the section sizes).
    func table(rotaryDims: Int, theta: Float, section: [Int], range: Range<Int>? = nil) -> (MLXArray, MLXArray) {
        let r = range ?? 0 ..< count
        let half = rotaryDims / 2
        let invFreq = MLXArray(
            stride(from: 0, to: rotaryDims, by: 2).map { 1 / pow(theta, Float($0) / Float(rotaryDims)) })
        func f(_ a: [Int32]) -> MLXArray {
            MLXArray(a[r].map { Float($0) }).expandedDimensions(axis: 1) * invFreq
        }
        var which = [Int32](repeating: 0, count: half)
        for i in 0 ..< half {
            if i % 3 == 1 && i < 3 * section[1] { which[i] = 1 }
            if i % 3 == 2 && i < 3 * section[2] { which[i] = 2 }
        }
        let sel = MLXArray(which)
        let freqs = MLX.where(sel .== 1, f(h), MLX.where(sel .== 2, f(w), f(t)))
        let emb = concatenated([freqs, freqs], axis: 1)
        return (MLX.cos(emb), MLX.sin(emb))
    }
}

// MARK: - The whole model

public final class Qwen35VL: Module {
    public let config: Qwen35Config

    @ModuleInfo(key: "vision") var vision: QwenVisionTower
    @ModuleInfo(key: "lm") var lm: TextModel

    init(_ config: Qwen35Config) {
        self.config = config
        _vision.wrappedValue = QwenVisionTower(config.vision)
        _lm.wrappedValue = TextModel(config.text)
    }

    /// Loads `config.json` and `model.safetensors` written by tools/export_qwen.py.
    public static func load(directory: URL) throws -> Qwen35VL {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let model = Qwen35VL(try JSONDecoder().decode(Qwen35Config.self, from: data))
        let weights = try loadArrays(url: directory.appendingPathComponent("model.safetensors"))
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)
        return model
    }

    var rotaryDims: Int { Int(Float(config.text.headDim) * config.text.partialRotaryFactor) }

    func makeCaches() -> [LayerCache] {
        lm.layers.map { $0.isLinear ? .linear(LinearCache()) : .attention(AttentionCache()) }
    }

    /// Token embeddings [1, L, H] with the vision placeholders (image or video pads) replaced by `features` [N, H]
    /// in order.
    func inputEmbeddings(tokenIDs: [Int32], features: MLXArray?) -> MLXArray {
        var emb = lm.embedTokens(MLXArray(tokenIDs))
        let pads = Set([Int32(config.imageTokenID), Int32(config.videoTokenID)])
        let positions = tokenIDs.enumerated().filter { pads.contains($0.element) }.map { Int32($0.offset) }
        precondition(positions.count == (features?.dim(0) ?? 0),
                     "\(positions.count) vision tokens, \(features?.dim(0) ?? 0) vision features")
        if let features, !positions.isEmpty {
            emb[MLXArray(positions)] = features.asType(emb.dtype)
        }
        return emb.expandedDimensions(axis: 0)
    }

    /// Runs `embeddings` [1, L, H] at `positions` through the language model in chunks of `chunk` tokens, advancing
    /// `caches`; returns the final-normed hidden state of the last token [1, 1, H].
    func prefill(_ embeddings: MLXArray, positions: MRopePositions, caches: [LayerCache], chunk: Int = 1024)
        -> MLXArray
    {
        let n = embeddings.dim(1)
        var last: MLXArray?
        var start = 0
        while start < n {
            let end = min(n, start + chunk)
            let (cos, sin) = positions.table(
                rotaryDims: rotaryDims, theta: config.text.ropeTheta, section: config.mropeSection,
                range: start ..< end)
            let h = lm(embeddings: embeddings[0..., start ..< end, 0...], caches: caches,
                       rotary: .table(cos: cos, sin: sin))
            last = h[0..., (end - start - 1) ..< (end - start), 0...]
            if end < n { eval([last!] + Self.arrays(caches)) }  // bound the graph between chunks
            start = end
        }
        return last!
    }

    /// The arrays a cache holds (to evaluate them).
    static func arrays(_ caches: [LayerCache]) -> [MLXArray] {
        caches.flatMap { c -> [MLXArray] in
            switch c {
            case .attention(let a): return [a.keys, a.values].compactMap { $0 }
            case .linear(let l): return [l.conv, l.recurrent].compactMap { $0 }
            }
        }
    }

    /// One text token at position `position` [1, 1, H] -> final-normed hidden [1, 1, H].
    func step(_ token: MLXArray, position: Int, caches: [LayerCache]) -> MLXArray {
        lm(embeddings: lm.embedTokens(token.reshaped([1])).expandedDimensions(axis: 0), caches: caches,
           rotary: .start(position))
    }

    func logits(_ hidden: MLXArray) -> MLXArray {
        lm.logits(hidden)
    }
}
