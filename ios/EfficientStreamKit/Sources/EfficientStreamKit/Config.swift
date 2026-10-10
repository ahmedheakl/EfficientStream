// Settings of the pipeline (efficientstream/config.py); the defaults are the evaluated configuration and the
// exported `efficientstream_config.json` overrides them.

import Foundation

public struct StreamConfig: Codable, Sendable {
    /// Frames per second admitted to the store (one every 2 s).
    public var gridFps: Double = 0.5
    /// A stored frame is resized to the largest multiple-of-32 size under this many pixels.
    public var gridMaxPixels: Int = 256 * 32 * 32
    /// JPEG round trip of stored frames (nil = raw pixels).
    public var storeJpegQuality: Int? = 88
    /// At most this many stored frames; on overflow the frame in the densest region is evicted.
    public var storeBudget: Int = 83
    /// Thumbnails shown: the stored frames most similar to the question, in time order.
    public var retrieveK: Int = 16
    /// Thumbnails are letterboxed to a square of this side (one vision view, 64 visual tokens).
    public var thumbSide: Int = 448
    /// The latest frame keeps its aspect ratio with this long side (up to `latestMaxSlices` views).
    public var latestLongSide: Int = 896
    public var latestMaxSlices: Int = 9
    public var latestJpegQuality: Int? = 88
    /// "hms" (00:06:57) or "s" (417s).
    public var timestamps: String = "hms"
    /// Open-ended answers: greedy tokens after the forced "<answer>".
    public var maxNewTokens: Int = 128
    /// Embeddings rounded to float16 before the similarity, as in the evaluated index.
    public var roundEmbeddingsFp16: Bool = true

    enum CodingKeys: String, CodingKey {
        case gridFps = "grid_fps"
        case gridMaxPixels = "grid_max_pixels"
        case storeJpegQuality = "store_jpeg_quality"
        case storeBudget = "store_budget"
        case retrieveK = "retrieve_k"
        case thumbSide = "thumb_side"
        case latestLongSide = "latest_long_side"
        case latestMaxSlices = "latest_max_slices"
        case latestJpegQuality = "latest_jpeg_quality"
        case timestamps
        case maxNewTokens = "max_new_tokens"
        case roundEmbeddingsFp16 = "round_embeddings_fp16"
    }

    public init() {}

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let d = StreamConfig()
        gridFps = try c.decodeIfPresent(Double.self, forKey: .gridFps) ?? d.gridFps
        gridMaxPixels = try c.decodeIfPresent(Int.self, forKey: .gridMaxPixels) ?? d.gridMaxPixels
        storeJpegQuality = c.contains(.storeJpegQuality)
            ? try c.decodeIfPresent(Int.self, forKey: .storeJpegQuality) : d.storeJpegQuality
        storeBudget = try c.decodeIfPresent(Int.self, forKey: .storeBudget) ?? d.storeBudget
        retrieveK = try c.decodeIfPresent(Int.self, forKey: .retrieveK) ?? d.retrieveK
        thumbSide = try c.decodeIfPresent(Int.self, forKey: .thumbSide) ?? d.thumbSide
        latestLongSide = try c.decodeIfPresent(Int.self, forKey: .latestLongSide) ?? d.latestLongSide
        latestMaxSlices = try c.decodeIfPresent(Int.self, forKey: .latestMaxSlices) ?? d.latestMaxSlices
        latestJpegQuality = c.contains(.latestJpegQuality)
            ? try c.decodeIfPresent(Int.self, forKey: .latestJpegQuality) : d.latestJpegQuality
        timestamps = try c.decodeIfPresent(String.self, forKey: .timestamps) ?? d.timestamps
        maxNewTokens = try c.decodeIfPresent(Int.self, forKey: .maxNewTokens) ?? d.maxNewTokens
        roundEmbeddingsFp16 = try c.decodeIfPresent(Bool.self, forKey: .roundEmbeddingsFp16) ?? d.roundEmbeddingsFp16
    }

    public static func load(_ url: URL) throws -> StreamConfig {
        try JSONDecoder().decode(StreamConfig.self, from: Data(contentsOf: url))
    }
}
