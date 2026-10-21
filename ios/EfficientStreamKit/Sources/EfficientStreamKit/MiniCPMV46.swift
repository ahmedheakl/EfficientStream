// MiniCPM-V-4.6 on MLX: the SigLIP vision tower with the window-attention merger inserted after layer
// `insertLayerID`, the 2x2 downsample merger into the language model's space, and the Qwen3.5 text model
// (gated-delta linear attention + gated full attention).
//
// Mirrors transformers' `modeling_minicpmv4_6.py` / `qwen3_5` (the evaluated PyTorch path). Weights are the
// checkpoint's, renamed and re-laid out by `tools/export_ios.py` (patch embedding as a linear layer over
// (row, column, channel) patches, depthwise conv weights as [channels, kernel, 1], RMSNorm weights with the
// checkpoint's +1 offset folded in, as float32).

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct MiniCPMVConfig: Codable, Sendable {
    public struct Vision: Codable, Sendable {
        public var hiddenSize: Int
        public var intermediateSize: Int
        public var numHiddenLayers: Int
        public var numAttentionHeads: Int
        public var patchSize: Int
        public var imageSize: Int
        public var layerNormEps: Float

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case numHiddenLayers = "num_hidden_layers"
            case numAttentionHeads = "num_attention_heads"
            case patchSize = "patch_size"
            case imageSize = "image_size"
            case layerNormEps = "layer_norm_eps"
        }
    }

    public struct Text: Codable, Sendable {
        public var hiddenSize: Int
        public var intermediateSize: Int
        public var numHiddenLayers: Int
        public var numAttentionHeads: Int
        public var numKeyValueHeads: Int
        public var headDim: Int
        public var rmsNormEps: Float
        public var vocabSize: Int
        public var linearNumKeyHeads: Int
        public var linearNumValueHeads: Int
        public var linearKeyHeadDim: Int
        public var linearValueHeadDim: Int
        public var linearConvKernelDim: Int
        public var layerTypes: [String]
        public var ropeTheta: Float
        public var partialRotaryFactor: Float

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case numHiddenLayers = "num_hidden_layers"
            case numAttentionHeads = "num_attention_heads"
            case numKeyValueHeads = "num_key_value_heads"
            case headDim = "head_dim"
            case rmsNormEps = "rms_norm_eps"
            case vocabSize = "vocab_size"
            case linearNumKeyHeads = "linear_num_key_heads"
            case linearNumValueHeads = "linear_num_value_heads"
            case linearKeyHeadDim = "linear_key_head_dim"
            case linearValueHeadDim = "linear_value_head_dim"
            case linearConvKernelDim = "linear_conv_kernel_dim"
            case layerTypes = "layer_types"
            case ropeTheta = "rope_theta"
            case partialRotaryFactor = "partial_rotary_factor"
        }
    }

    public struct Quantization: Codable, Sendable {
        public var bits: Int
        public var groupSize: Int
        enum CodingKeys: String, CodingKey {
            case bits
            case groupSize = "group_size"
        }
    }

    public var vision: Vision
    public var text: Text
    public var insertLayerID: Int
    public var imageTokenID: Int
    /// Present when the export quantized the language model's linear layers.
    public var quantization: Quantization?

    enum CodingKeys: String, CodingKey {
        case vision = "vision_config"
        case text = "text_config"
        case insertLayerID = "insert_layer_id"
        case imageTokenID = "image_token_id"
        case quantization
    }
}

// MARK: - Vision tower

final class VisionAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(dim: Int, heads: Int) {
        numHeads = heads
        headDim = dim / heads
        scale = pow(Float(headDim), -0.5)
        _qProj.wrappedValue = Linear(dim, dim)
        _kProj.wrappedValue = Linear(dim, dim)
        _vProj.wrappedValue = Linear(dim, dim)
        _outProj.wrappedValue = Linear(dim, dim)
    }

    /// Full bidirectional attention within each row of the batch: x [B, L, D].
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        let q = qProj(x).reshaped(b, l, numHeads, headDim).transposed(0, 2, 1, 3)
        let k = kProj(x).reshaped(b, l, numHeads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(b, l, numHeads, headDim).transposed(0, 2, 1, 3)
        let o = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: .none)
        return outProj(o.transposed(0, 2, 1, 3).reshaped(b, l, -1))
    }
}

final class VisionMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(_ c: MiniCPMVConfig.Vision) {
        _fc1.wrappedValue = Linear(c.hiddenSize, c.intermediateSize)
        _fc2.wrappedValue = Linear(c.intermediateSize, c.hiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(geluApproximate(fc1(x)))  // gelu_pytorch_tanh
    }
}

final class VisionEncoderLayer: Module {
    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
    @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm
    @ModuleInfo(key: "self_attn") var selfAttn: VisionAttention
    @ModuleInfo(key: "mlp") var mlp: VisionMLP

    init(_ c: MiniCPMVConfig.Vision) {
        _layerNorm1.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEps)
        _layerNorm2.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEps)
        _selfAttn.wrappedValue = VisionAttention(dim: c.hiddenSize, heads: c.numAttentionHeads)
        _mlp.wrappedValue = VisionMLP(c)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = x + selfAttn(layerNorm1(x))
        return h + mlp(layerNorm2(h))
    }
}

/// Window-attention merger inserted inside the vision tower: attention within each 2x2 window, then the four
/// tokens of a window are fused into one (residual: their mean). Grid (h, w) -> (h/2, w/2), row-major.
final class ViTWindowMerger: Module {
    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
    @ModuleInfo(key: "self_attn") var selfAttn: VisionAttention
    @ModuleInfo(key: "pre_norm") var preNorm: LayerNorm
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(_ c: MiniCPMVConfig.Vision) {
        let d = c.hiddenSize
        _layerNorm1.wrappedValue = LayerNorm(dimensions: d, eps: c.layerNormEps)
        _selfAttn.wrappedValue = VisionAttention(dim: d, heads: c.numAttentionHeads)
        _preNorm.wrappedValue = LayerNorm(dimensions: 4 * d, eps: c.layerNormEps)
        _linear1.wrappedValue = Linear(4 * d, 4 * c.intermediateSize)
        _linear2.wrappedValue = Linear(4 * c.intermediateSize, d)
    }

    /// x [B, h*w, D] (row-major grid) -> [B, (h/2)*(w/2), D]
    func callAsFunction(_ x: MLXArray, h: Int, w: Int) -> MLXArray {
        let (b, d) = (x.dim(0), x.dim(2))
        let n = (h / 2) * (w / 2)
        let windows = x.reshaped(b, h / 2, 2, w / 2, 2, d).transposed(0, 1, 3, 2, 4, 5)
            .reshaped(b * n, 4, d)
        let attended = windows + selfAttn(layerNorm1(windows))
        let residual = attended.mean(axis: 1)
        var y = preNorm(attended.reshaped(b * n, 4 * d))
        y = linear2(geluApproximate(linear1(y)))  // gelu_pytorch_tanh
        return (y + residual).reshaped(b, n, d)
    }
}

/// The final 2x2 downsample into the language model's hidden size (one MLP, `merger_times` = 1).
final class DownsampleMLP: Module {
    @ModuleInfo(key: "pre_norm") var preNorm: LayerNorm
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(visionHidden: Int, outSize: Int) {
        let d = 4 * visionHidden
        _preNorm.wrappedValue = LayerNorm(dimensions: d, eps: 1e-6)
        _linear1.wrappedValue = Linear(d, d)
        _linear2.wrappedValue = Linear(d, outSize)
    }

    /// x [B, h*w, D] -> [B, (h/2)*(w/2), outSize]
    func callAsFunction(_ x: MLXArray, h: Int, w: Int) -> MLXArray {
        let (b, d) = (x.dim(0), x.dim(2))
        let n = (h / 2) * (w / 2)
        let y = x.reshaped(b, h / 2, 2, w / 2, 2, d).transposed(0, 1, 3, 2, 4, 5).reshaped(b, n, 4 * d)
        return linear2(gelu(linear1(preNorm(y))))  // nn.GELU() (exact)
    }
}

final class VisionTower: Module {
    let config: MiniCPMVConfig.Vision
    let insertLayerID: Int

    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Linear
    @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding
    let layers: [VisionEncoderLayer]
    @ModuleInfo(key: "post_layernorm") var postLayerNorm: LayerNorm
    @ModuleInfo(key: "vit_merger") var vitMerger: ViTWindowMerger

    init(_ c: MiniCPMVConfig.Vision, insertLayerID: Int) {
        config = c
        self.insertLayerID = insertLayerID
        let side = c.imageSize / c.patchSize
        _patchEmbedding.wrappedValue = Linear(c.patchSize * c.patchSize * 3, c.hiddenSize)
        _positionEmbedding.wrappedValue = Embedding(embeddingCount: side * side, dimensions: c.hiddenSize)
        layers = (0 ..< c.numHiddenLayers).map { _ in VisionEncoderLayer(c) }
        _postLayerNorm.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEps)
        _vitMerger.wrappedValue = ViTWindowMerger(c)
    }

    /// pixels [B, h*14, w*14, 3] (normalized) -> [B, (h/2)*(w/2), D] after the post-layernorm.
    func callAsFunction(_ pixels: MLXArray, h: Int, w: Int) -> MLXArray {
        let b = pixels.dim(0)
        let p = config.patchSize
        let patches = pixels.reshaped(b, h, p, w, p, 3).transposed(0, 1, 3, 2, 4, 5)
            .reshaped(b, h * w, p * p * 3)
        let side = config.imageSize / p
        let pos = MLXArray(nearestPositionIDs(h: h, w: w, side: side).map { Int32($0) })
        var x = patchEmbedding(patches.asType(patchEmbedding.weight.dtype)) + positionEmbedding(pos)
        var (gh, gw) = (h, w)
        for (i, layer) in layers.enumerated() {
            x = layer(x)
            if i == insertLayerID {
                x = vitMerger(x, h: gh, w: gw)
                (gh, gw) = (gh / 2, gw / 2)
            }
        }
        return postLayerNorm(x)
    }
}

/// transformers' `get_vision_nearest_position_ids`: bucketize(i / n, arange(1/side, 1, 1/side), right=True),
/// in float32 like the reference.
func nearestPositionIDs(h: Int, w: Int, side: Int) -> [Int] {
    let step = 1.0 / Double(side)
    var boundaries: [Float] = []
    var k = 1
    while Double(k) * step < 1.0 {
        boundaries.append(Float(Double(k) * step))
        k += 1
    }
    func bucket(_ i: Int, _ n: Int) -> Int {
        let x = Float(i) / Float(n)
        var c = 0
        for bd in boundaries where bd <= x { c += 1 }
        return c
    }
    let bh = (0 ..< h).map { bucket($0, h) }
    let bw = (0 ..< w).map { bucket($0, w) }
    var out: [Int] = []
    out.reserveCapacity(h * w)
    for r in bh { for c in bw { out.append(r * side + c) } }
    return out
}

// MARK: - Qwen3.5 text model

final class TextAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float
    let rope: RoPE

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: FP32RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: FP32RMSNorm

    init(_ c: MiniCPMVConfig.Text) {
        numHeads = c.numAttentionHeads
        numKVHeads = c.numKeyValueHeads
        headDim = c.headDim
        scale = pow(Float(headDim), -0.5)
        rope = RoPE(
            dimensions: Int(Float(headDim) * c.partialRotaryFactor), traditional: false,
            base: c.ropeTheta)
        _qProj.wrappedValue = Linear(c.hiddenSize, numHeads * headDim * 2, bias: false)
        _kProj.wrappedValue = Linear(c.hiddenSize, numKVHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(c.hiddenSize, numKVHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(numHeads * headDim, c.hiddenSize, bias: false)
        _qNorm.wrappedValue = FP32RMSNorm(dimensions: headDim, eps: c.rmsNormEps)
        _kNorm.wrappedValue = FP32RMSNorm(dimensions: headDim, eps: c.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, cache: AttentionCache, rotary: TextRotary? = nil) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        let qg = qProj(x).reshaped(b, l, numHeads, 2 * headDim)
        let parts = MLX.split(qg, parts: 2, axis: -1)
        let gate = parts[1].reshaped(b, l, -1)
        var q = qNorm(parts[0]).transposed(0, 2, 1, 3)
        var k = kNorm(kProj(x).reshaped(b, l, numKVHeads, headDim)).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(b, l, numKVHeads, headDim).transposed(0, 2, 1, 3)
        switch rotary {
        case nil:
            q = rope(q, offset: cache.offset)
            k = rope(k, offset: cache.offset)
        case .start(let p):
            q = rope(q, offset: p)
            k = rope(k, offset: p)
        case .table(let cos, let sin):
            q = applyRotaryTable(q, cos: cos, sin: sin)
            k = applyRotaryTable(k, cos: cos, sin: sin)
        }
        let (keys, values) = cache.append(keys: k, values: v)
        if cache.captureQuery { cache.lastQuery = q[0..., 0..., (l - 1) ..< l, 0...] }
        let o = MLXFast.scaledDotProductAttention(
            queries: q, keys: keys, values: values, scale: scale, mask: l > 1 ? .causal : .none)
        return oProj(o.transposed(0, 2, 1, 3).reshaped(b, l, -1) * sigmoid(gate))
    }
}

/// Positions for the text model's rotary embedding. `nil` (MiniCPM-V-4.6): positions continue from the cache.
/// `.start(p)`: consecutive positions from `p` (Qwen3.5-VL text after the prompt: M-RoPE positions do not follow the
/// cache length). `.table`: per-token cos/sin [L, rotary dims], the interleaved M-RoPE of a multimodal prompt.
enum TextRotary {
    case start(Int)
    case table(cos: MLXArray, sin: MLXArray)
}

/// Rotate-half rotary over the first `cos.dim(-1)` dims of x [B, H, L, D] (transformers' apply_rotary_pos_emb with
/// partial rotary; cos/sin cast to x's dtype as there).
func applyRotaryTable(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
    let r = cos.dim(-1)
    let half = r / 2
    let rot = x[0..., 0..., 0..., 0 ..< r]
    let x1 = rot[0..., 0..., 0..., 0 ..< half]
    let x2 = rot[0..., 0..., 0..., half ..< r]
    let out = rot * cos.asType(x.dtype) + concatenated([-x2, x1], axis: -1) * sin.asType(x.dtype)
    return x.dim(-1) == r ? out : concatenated([out, x[0..., 0..., 0..., r...]], axis: -1)
}

/// Qwen3.5's RMSNorm, (1 + w) * x / rms(x) computed in float32 (the export stores 1 + w as float32).
final class FP32RMSNorm: Module {
    let eps: Float
    let weight: MLXArray

    init(dimensions: Int, eps: Float) {
        self.eps = eps
        weight = MLXArray.ones([dimensions], dtype: .float32)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x.asType(.float32), weight: weight, eps: eps).asType(x.dtype)
    }
}

final class GatedRMSNorm: Module {
    let eps: Float
    let weight: MLXArray

    init(dimensions: Int, eps: Float) {
        self.eps = eps
        weight = MLXArray.ones([dimensions])
    }

    func callAsFunction(_ x: MLXArray, gate: MLXArray) -> MLXArray {
        let y = MLXFast.rmsNorm(x, weight: weight, eps: eps)
        return (silu(gate.asType(.float32)) * y.asType(.float32)).asType(x.dtype)
    }
}

final class GatedDeltaNet: Module {
    let numKHeads: Int
    let numVHeads: Int
    let headKDim: Int
    let headVDim: Int
    let keyDim: Int
    let valueDim: Int
    let convKernel: Int

    @ModuleInfo(key: "in_proj_qkv") var inProjQKV: Linear
    @ModuleInfo(key: "in_proj_z") var inProjZ: Linear
    @ModuleInfo(key: "in_proj_b") var inProjB: Linear
    @ModuleInfo(key: "in_proj_a") var inProjA: Linear
    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ModuleInfo(key: "norm") var norm: GatedRMSNorm
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ c: MiniCPMVConfig.Text) {
        numKHeads = c.linearNumKeyHeads
        numVHeads = c.linearNumValueHeads
        headKDim = c.linearKeyHeadDim
        headVDim = c.linearValueHeadDim
        keyDim = numKHeads * headKDim
        valueDim = numVHeads * headVDim
        convKernel = c.linearConvKernelDim
        let convDim = 2 * keyDim + valueDim
        _inProjQKV.wrappedValue = Linear(c.hiddenSize, convDim, bias: false)
        _inProjZ.wrappedValue = Linear(c.hiddenSize, valueDim, bias: false)
        _inProjB.wrappedValue = Linear(c.hiddenSize, numVHeads, bias: false)
        _inProjA.wrappedValue = Linear(c.hiddenSize, numVHeads, bias: false)
        _conv1d.wrappedValue = Conv1d(
            inputChannels: convDim, outputChannels: convDim, kernelSize: convKernel, stride: 1,
            padding: 0, dilation: 1, groups: convDim, bias: false)
        _dtBias.wrappedValue = MLXArray.ones([numVHeads])
        _aLog.wrappedValue = MLXArray.zeros([numVHeads])
        _norm.wrappedValue = GatedRMSNorm(dimensions: headVDim, eps: c.rmsNormEps)
        _outProj.wrappedValue = Linear(valueDim, c.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray, cache: LinearCache) -> MLXArray {
        let (b, s) = (x.dim(0), x.dim(1))
        let qkv = inProjQKV(x)
        let z = inProjZ(x).reshaped(b, s, numVHeads, headVDim)
        let beta = inProjB(x)
        let a = inProjA(x)

        let convState =
            cache.conv ?? MLXArray.zeros([b, convKernel - 1, qkv.dim(-1)], dtype: qkv.dtype)
        let convInput = concatenated([convState, qkv], axis: 1)
        cache.conv = convInput[0..., (convInput.dim(1) - (convKernel - 1))..., 0...]
        let convOut = silu(conv1d(convInput))

        let split = MLX.split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)
        let q = split[0].reshaped(b, s, numKHeads, headKDim)
        let k = split[1].reshaped(b, s, numKHeads, headKDim)
        let v = split[2].reshaped(b, s, numVHeads, headVDim)

        // l2-normalize q and k (reference eps 1e-6 on sum(x^2)); fold the head_dim^-0.5 readout scale into q
        let invScale = pow(Float(headKDim), -0.5)
        let qkEps = 1e-6 / Float(headKDim)
        let qn = MLXArray(invScale * invScale).asType(q.dtype)
            * MLXFast.rmsNorm(q, weight: MLXArray.mlxNone, eps: qkEps)
        let kn = MLXArray(invScale).asType(k.dtype)
            * MLXFast.rmsNorm(k, weight: MLXArray.mlxNone, eps: qkEps)

        let (out, state) = gatedDeltaUpdate(
            q: qn, k: kn, v: v, a: a, b: beta, aLog: aLog, dtBias: dtBias, state: cache.recurrent)
        cache.recurrent = state
        return outProj(norm(out, gate: z).reshaped(b, s, -1))
    }
}

final class TextMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ c: MiniCPMVConfig.Text) {
        _gateProj.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(c.intermediateSize, c.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

final class TextDecoderLayer: Module {
    let isLinear: Bool

    @ModuleInfo(key: "linear_attn") var linearAttn: GatedDeltaNet?
    @ModuleInfo(key: "self_attn") var selfAttn: TextAttention?
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: FP32RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: FP32RMSNorm
    @ModuleInfo(key: "mlp") var mlp: TextMLP

    init(_ c: MiniCPMVConfig.Text, index: Int) {
        isLinear = c.layerTypes[index] == "linear_attention"
        if isLinear {
            _linearAttn.wrappedValue = GatedDeltaNet(c)
        } else {
            _selfAttn.wrappedValue = TextAttention(c)
        }
        _inputLayerNorm.wrappedValue = FP32RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = FP32RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _mlp.wrappedValue = TextMLP(c)
    }

    func callAsFunction(_ x: MLXArray, cache: LayerCache, rotary: TextRotary? = nil) -> MLXArray {
        let mix = { () -> MLXArray in
            switch cache {
            case .linear(let c): return self.linearAttn!(self.inputLayerNorm(x), cache: c)
            case .attention(let c): return self.selfAttn!(self.inputLayerNorm(x), cache: c, rotary: rotary)
            }
        }
        if let p = LayerProfiler.active {
            let h = x + p.time(isLinear ? "gated delta layers" : "full attention layers", mix)
            return p.time("MLPs") { h + self.mlp(self.postAttentionLayerNorm(h)) }
        }
        let h = x + mix()
        return h + mlp(postAttentionLayerNorm(h))
    }
}

final class TextModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [TextDecoderLayer]
    @ModuleInfo(key: "norm") var norm: FP32RMSNorm

    init(_ c: MiniCPMVConfig.Text) {
        _embedTokens.wrappedValue = Embedding(embeddingCount: c.vocabSize, dimensions: c.hiddenSize)
        layers = (0 ..< c.numHiddenLayers).map { TextDecoderLayer(c, index: $0) }
        _norm.wrappedValue = FP32RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
    }

    /// Final-normed hidden states for `embeddings` [1, L, H], advancing `caches`.
    func callAsFunction(embeddings: MLXArray, caches: [LayerCache], rotary: TextRotary? = nil) -> MLXArray {
        var h = embeddings
        for (layer, cache) in zip(layers, caches) {
            h = layer(h, cache: cache, rotary: rotary)
        }
        return norm(h)
    }

    /// Tied output head.
    func logits(_ hidden: MLXArray) -> MLXArray {
        if let p = LayerProfiler.active { return p.time("output head") { embedTokens.asLinear(hidden) } }
        return embedTokens.asLinear(hidden)
    }
}

/// Times the text model's parts while `active` is set (benchmarks only: it evaluates after every part, which removes
/// the GPU's overlap between them, so the shares matter more than the sum).
public final class LayerProfiler: @unchecked Sendable {
    nonisolated(unsafe) public static var active: LayerProfiler?
    public private(set) var seconds: [String: Double] = [:]

    public init() {}

    func time(_ key: String, _ body: () -> MLXArray) -> MLXArray {
        let t0 = Date()
        let x = body()
        eval(x)
        seconds[key, default: 0] += Date().timeIntervalSince(t0)
        return x
    }
}

// MARK: - Caches

/// Keys and values in buffers grown `step` tokens at a time and written in place (mlx-lm's KVCacheSimple): an append
/// copies only the new tokens, not the whole cache. MLX arrays stay values: while another cache object holds the same
/// buffer (a fork), the next write copies it instead of changing what the fork sees.
final class AttentionCache {
    var keys: MLXArray?
    var values: MLXArray?
    var offset = 0
    static let step = 1024
    /// While set, each call keeps its last token's query (rotated) [1, heads, 1, headDim] (QwenStreamer `.compress`:
    /// how much the token that decides a second attends to each frame block).
    var captureQuery = false
    var lastQuery: MLXArray?

    func append(keys k: MLXArray, values v: MLXArray) -> (MLXArray, MLXArray) {
        let previous = offset
        let n = k.dim(2)
        if keys == nil || previous + n > keys!.dim(2) {
            let grow = (n + Self.step - 1) / Self.step * Self.step
            let zk = MLXArray.zeros([k.dim(0), k.dim(1), grow, k.dim(3)], dtype: k.dtype)
            let zv = MLXArray.zeros([v.dim(0), v.dim(1), grow, v.dim(3)], dtype: v.dtype)
            if let k0 = keys, let v0 = values {
                keys = concatenated([k0[.ellipsis, ..<previous, 0...], zk], axis: 2)
                values = concatenated([v0[.ellipsis, ..<previous, 0...], zv], axis: 2)
            } else {
                keys = zk
                values = zv
            }
        }
        offset += n
        keys![.ellipsis, previous ..< offset, 0...] = k
        values![.ellipsis, previous ..< offset, 0...] = v
        return (keys![.ellipsis, ..<offset, 0...], values![.ellipsis, ..<offset, 0...])
    }
    /// The share of `query`'s attention [1, heads, 1, headDim] (softmax over the cached tokens, averaged over the
    /// heads) that falls on each of `spans`.
    func attentionShare(_ query: MLXArray, spans: [Range<Int>]) -> [Float] {
        guard let k0 = keys, offset > 0 else { return spans.map { _ in 0 } }
        let k = k0[.ellipsis, ..<offset, 0...]
        let (heads, kvHeads, d) = (query.dim(1), k.dim(1), query.dim(3))
        let q = query.reshaped(1, kvHeads, heads / kvHeads, d).asType(.float32)
        let scores = matmul(q, k.asType(.float32).transposed(0, 1, 3, 2)) * pow(Float(d), -0.5)
        let p = softmax(scores, axis: -1).mean(axes: [0, 1, 2])
        let shares = spans.map { $0.isEmpty ? MLXArray(Float(0)) : p[$0.lowerBound ..< $0.upperBound].sum() }
        let out = stacked(shares)
        eval(out)
        return out.asArray(Float.self)
    }

    /// Drops the tokens [a, b) (later ones move down).
    func evict(_ a: Int, _ b: Int) {
        guard let k0 = keys, let v0 = values, a < b, b <= offset else { return }
        keys = concatenated([k0[.ellipsis, ..<a, 0...], k0[.ellipsis, b ..< offset, 0...]], axis: 2)
        values = concatenated([v0[.ellipsis, ..<a, 0...], v0[.ellipsis, b ..< offset, 0...]], axis: 2)
        offset -= b - a
    }
}

final class LinearCache {
    var conv: MLXArray?
    var recurrent: MLXArray?
}

enum LayerCache {
    case linear(LinearCache)
    case attention(AttentionCache)
}

// MARK: - The whole model

public final class MiniCPMV46: Module {
    public let config: MiniCPMVConfig

    @ModuleInfo(key: "vision") var vision: VisionTower
    @ModuleInfo(key: "merger") var merger: DownsampleMLP
    @ModuleInfo(key: "lm") var lm: TextModel

    init(_ config: MiniCPMVConfig) {
        self.config = config
        _vision.wrappedValue = VisionTower(config.vision, insertLayerID: config.insertLayerID)
        _merger.wrappedValue = DownsampleMLP(
            visionHidden: config.vision.hiddenSize, outSize: config.text.hiddenSize)
        _lm.wrappedValue = TextModel(config.text)
    }

    /// Loads `config.json` and `model.safetensors` written by tools/export_ios.py.
    public static func load(directory: URL) throws -> MiniCPMV46 {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(MiniCPMVConfig.self, from: data)
        let model = MiniCPMV46(config)
        if let q = config.quantization {
            // only the language model's linear layers are quantized (see export_ios.py)
            quantize(model: model.lm, groupSize: q.groupSize, bits: q.bits) { path, module in
                guard module is Linear else { return false }
                return !(path.hasSuffix("in_proj_a") || path.hasSuffix("in_proj_b"))
            }
        }
        let weights = try loadArrays(url: directory.appendingPathComponent("model.safetensors"))
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)
        return model
    }

    /// Vision features of views that share one patch grid: pixels [B, h*14, w*14, 3] -> [B, h*w/16, hidden].
    public func imageFeatures(_ pixels: MLXArray, gridH: Int, gridW: Int) -> MLXArray {
        let x = vision(pixels, h: gridH, w: gridW)
        return merger(x, h: gridH / 2, w: gridW / 2)
    }

    func makeCaches() -> [LayerCache] {
        lm.layers.map { $0.isLinear ? .linear(LinearCache()) : .attention(AttentionCache()) }
    }

    /// Token embeddings with the image placeholders replaced by `imageFeatures` [N, hidden] (in order).
    public func inputEmbeddings(tokenIDs: [Int32], imageFeatures: MLXArray) -> MLXArray {
        let ids = MLXArray(tokenIDs)
        var emb = lm.embedTokens(ids)
        let positions = tokenIDs.enumerated().filter { $0.element == Int32(config.imageTokenID) }
            .map { Int32($0.offset) }
        precondition(
            positions.count == imageFeatures.dim(0),
            "\(positions.count) image tokens, \(imageFeatures.dim(0)) image features")
        if !positions.isEmpty {
            emb[MLXArray(positions)] = imageFeatures.asType(emb.dtype)
        }
        return emb.expandedDimensions(axis: 0)
    }
}
