// The bounded store of past frames (efficientstream/frames.py, BoundedStore).

import Foundation
import MLX

public struct StoredFrame: @unchecked Sendable {
    public let time: Double
    /// The stored (grid-size) frame.
    public let image: RGBImage
    /// Unit-norm MobileCLIP2-S0 image embedding (rounded to float16 when configured).
    public let embedding: [Float]
    /// The frame's thumbnail (letterboxed, one view) through the vision tower, computed once when the frame
    /// enters the store: [64, hidden]. The reference computes the same features at question time.
    public let thumbnail: EncodedImage?
}

/// Frames arrive in time order. When the store holds more than `budget` frames, the one whose removal leaves the
/// smallest gap between its neighbours (the densest region) is evicted. The first and the newest frame are never
/// evicted, so the store always spans the whole stream.
public struct BoundedStore: Sendable {
    public let budget: Int
    public private(set) var frames: [StoredFrame] = []

    public init(budget: Int) { self.budget = budget }

    public enum StoreError: Error { case outOfOrder(Double, Double) }

    /// Appends a frame; returns the evicted frame, if any.
    @discardableResult
    public mutating func add(_ frame: StoredFrame) throws -> StoredFrame? {
        if let last = frames.last, frame.time < last.time { throw StoreError.outOfOrder(frame.time, last.time) }
        frames.append(frame)
        if frames.count <= budget { return nil }
        let hi = frames.count - 1  // the newest frame is not evictable
        if hi <= 1 { return frames.remove(at: 1) }
        var best = 1
        var bestGap = Double.infinity
        for j in 1 ..< hi {
            let gap = frames[j + 1].time - frames[j - 1].time
            if gap < bestGap {  // np.argmin: first minimum
                bestGap = gap
                best = j
            }
        }
        return frames.remove(at: best)
    }

    /// Stored frames strictly older than `t` (the latest frame is shown separately).
    public func before(_ t: Double) -> [StoredFrame] { frames.filter { $0.time < t - 1e-6 } }

    public var count: Int { frames.count }
}

/// Indices of the k rows most similar to `query` (stable on ties), in ascending (time) order.
public func topKInTimeOrder(_ embeddings: [[Float]], query: [Float], k: Int) -> [Int] {
    if embeddings.count <= k { return Array(embeddings.indices) }
    let sims = embeddings.map { e in zip(e, query).reduce(Float(0)) { $0 + $1.0 * $1.1 } }
    let order = sims.indices.sorted { sims[$0] > sims[$1] || (sims[$0] == sims[$1] && $0 < $1) }
    return order.prefix(k).sorted()
}
