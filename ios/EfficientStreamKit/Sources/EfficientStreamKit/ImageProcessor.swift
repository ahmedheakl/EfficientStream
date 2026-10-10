// The MiniCPM-V-4.6 image processor (transformers' MiniCPMV4_6ImageProcessor): an overview resized to about
// `scaleResolution`^2 pixels and, for larger images, a grid of slices of a refined resize. Each view's sides are
// multiples of 56 (patch 14 x two 2x2 merges), so a view of grid (h, w) patches gives h*w/16 tokens.

import Foundation
import MLX

public struct ImageView: Sendable {
    /// The resized view, 8-bit RGB, (gridW*14) x (gridH*14).
    public let image: RGBImage
    public let gridH: Int
    public let gridW: Int

    public init(image: RGBImage, gridH: Int, gridW: Int) {
        self.image = image
        self.gridH = gridH
        self.gridW = gridW
    }
}

public struct ProcessedImage: Sendable {
    public let views: [ImageView]  // overview first, then slices row by row
    public let sliceGrid: (rows: Int, cols: Int)?

    public func tokens(divisor: Int) -> [Int] { views.map { $0.gridH * $0.gridW / divisor } }
}

public enum MiniCPMImageProcessor {
    static let patch = 14

    /// Python `max(round(length / divisor) * divisor, divisor)` (round half to even).
    static func ensureDivide(_ length: Double, _ divisor: Int) -> Int {
        max(Int((length / Double(divisor)).rounded(.toNearestOrEven)) * divisor, divisor)
    }

    static func findBestResize(height: Double, width: Double, scaleResolution: Int, allowUpscale: Bool)
        -> (Int, Int)
    {
        var (h, w) = (height, width)
        let s = Double(scaleResolution)
        if h * w > s * s || allowUpscale {
            let aspect = w / h
            h = Double(Int(s / aspect.squareRoot()))
            w = Double(Int(h * aspect))
        }
        return (ensureDivide(h, patch * 4), ensureDivide(w, patch * 4))
    }

    static func refineSize(height: Int, width: Int, grid: (Int, Int), scaleResolution: Int) -> (Int, Int) {
        let (gy, gx) = grid
        let rw = ensureDivide(Double(width), gx)
        let rh = ensureDivide(Double(height), gy)
        let (bh, bw) = findBestResize(
            height: Double(rh) / Double(gy), width: Double(rw) / Double(gx), scaleResolution: scaleResolution,
            allowUpscale: true)
        return (bh * gy, bw * gx)
    }

    static func slicedGrid(height: Int, width: Int, maxSlices: Int, scaleResolution: Int) -> (Int, Int)? {
        let logRatio = log(Double(width) / Double(height))
        let ratio = Double(width * height) / Double(scaleResolution * scaleResolution)
        let multiple = min(Int(ratio.rounded(.up)), maxSlices)
        if multiple <= 1 { return nil }
        var best = (1, 1)
        var minError = Double.infinity
        for n in [multiple - 1, multiple, multiple + 1] where n != 1 && n <= maxSlices {
            for rows in 1 ... n where n % rows == 0 {
                let cols = n / rows
                let err = abs(logRatio - log(Double(cols) / Double(rows)))
                if err < minError {
                    best = (rows, cols)
                    minError = err
                } else if err == minError && rows > best.0 {
                    best = (rows, cols)
                }
            }
        }
        return best
    }

    public static func process(_ image: RGBImage, scaleResolution: Int, maxSlices: Int) -> ProcessedImage {
        let (h, w) = (image.height, image.width)
        let grid = slicedGrid(height: h, width: w, maxSlices: maxSlices, scaleResolution: scaleResolution)
        let (sh, sw) = findBestResize(
            height: Double(h), width: Double(w), scaleResolution: scaleResolution, allowUpscale: grid == nil)
        var views = [
            ImageView(
                image: image.resized(width: sw, height: sh, filter: .bicubic), gridH: sh / patch, gridW: sw / patch)
        ]
        if let grid {
            let (rh, rw) = refineSize(height: h, width: w, grid: grid, scaleResolution: scaleResolution)
            let refined = image.resized(width: rw, height: rh, filter: .bicubic)
            let (ph, pw) = (rh / grid.0, rw / grid.1)
            for r in 0 ..< grid.0 {
                for c in 0 ..< grid.1 {
                    views.append(
                        ImageView(
                            image: refined.cropped(x: c * pw, y: r * ph, width: pw, height: ph),
                            gridH: ph / patch, gridW: pw / patch))
                }
            }
        }
        return ProcessedImage(views: views, sliceGrid: grid.map { (rows: $0.0, cols: $0.1) })
    }

    /// Normalized pixels (mean 0.5 / std 0.5) of views that share one grid: [B, H, W, 3] float32.
    public static func pixels(_ views: [ImageView]) -> MLXArray {
        let (hh, ww) = (views[0].image.height, views[0].image.width)
        var buf = [Float](repeating: 0, count: views.count * hh * ww * 3)
        for (i, v) in views.enumerated() {
            precondition(v.image.height == hh && v.image.width == ww)
            let base = i * hh * ww * 3
            for j in 0 ..< hh * ww * 3 {
                buf[base + j] = (Float(v.image.pixels[j]) * Float(1.0 / 255.0) - 0.5) / 0.5
            }
        }
        return MLXArray(buf, [views.count, hh, ww, 3])
    }
}
