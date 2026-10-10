// Image handling of the pipeline (efficientstream/frames.py and the MiniCPM-V-4.6 image processor), on 8-bit RGB.
//
// `resized` reproduces Pillow's resampling (the reference resizes with Pillow and with torchvision's uint8
// antialiased kernels, which follow Pillow): separable passes, horizontal first, 22-bit fixed-point coefficients,
// rounding to 8 bits after each pass.

import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

public struct RGBImage: Sendable {
    public let width: Int
    public let height: Int
    /// Row-major RGB, 3 bytes per pixel.
    public var pixels: [UInt8]

    public init(width: Int, height: Int, pixels: [UInt8]) {
        precondition(pixels.count == width * height * 3)
        self.width = width
        self.height = height
        self.pixels = pixels
    }

    public init(width: Int, height: Int, fill: UInt8 = 0) {
        self.init(width: width, height: height, pixels: [UInt8](repeating: fill, count: width * height * 3))
    }
}

public enum ResampleFilter: Sendable {
    case bicubic, bilinear

    var support: Double { self == .bicubic ? 2.0 : 1.0 }

    func weight(_ x: Double) -> Double {
        let x = abs(x)
        switch self {
        case .bicubic:
            let a = -0.5
            if x < 1.0 { return ((a + 2.0) * x - (a + 3.0)) * x * x + 1 }
            if x < 2.0 { return (((x - 5) * x + 8) * x - 4) * a }
            return 0
        case .bilinear:
            return x < 1.0 ? 1.0 - x : 0
        }
    }
}

private let precisionBits = 32 - 8 - 2

/// Pillow's `precompute_coeffs` + `normalize_coeffs_8bpc` for one axis.
private func coefficients(inSize: Int, outSize: Int, filter: ResampleFilter)
    -> (bounds: [(Int, Int)], kernel: [Int32], kernelSize: Int)
{
    let scale = Double(inSize) / Double(outSize)
    let filterScale = max(scale, 1.0)
    let support = filter.support * filterScale
    let kernelSize = Int(ceil(support)) * 2 + 1
    var bounds: [(Int, Int)] = []
    var kernel = [Int32](repeating: 0, count: outSize * kernelSize)
    var k = [Double](repeating: 0, count: kernelSize)
    let ss = 1.0 / filterScale
    for xx in 0 ..< outSize {
        let center = (Double(xx) + 0.5) * scale
        var xmin = Int(center - support + 0.5)  // C truncation
        if xmin < 0 { xmin = 0 }
        var xmax = Int(center + support + 0.5)
        if xmax > inSize { xmax = inSize }
        xmax -= xmin
        var ww = 0.0
        for x in 0 ..< xmax {
            let w = filter.weight((Double(x + xmin) - center + 0.5) * ss)
            k[x] = w
            ww += w
        }
        for x in 0 ..< kernelSize {
            var w = x < xmax ? k[x] : 0
            if ww != 0 && x < xmax { w /= ww }
            let fixed = w < 0 ? Int32(-0.5 + w * Double(1 << precisionBits))
                : Int32(0.5 + w * Double(1 << precisionBits))
            kernel[xx * kernelSize + x] = fixed
        }
        bounds.append((xmin, xmax))
    }
    return (bounds, kernel, kernelSize)
}

@inline(__always) private func clip8(_ v: Int64) -> UInt8 {
    let s = v >> Int64(precisionBits)
    return s < 0 ? 0 : (s > 255 ? 255 : UInt8(s))
}

extension RGBImage {
    /// Pillow `Image.resize((width, height), filter)` (no reducing gap).
    public func resized(width w: Int, height h: Int, filter: ResampleFilter = .bicubic) -> RGBImage {
        if w == width && h == height { return self }
        var img = self
        if w != width { img = img.resampledHorizontally(to: w, filter: filter) }
        if h != height { img = img.resampledVertically(to: h, filter: filter) }
        return img
    }

    private func resampledHorizontally(to outW: Int, filter: ResampleFilter) -> RGBImage {
        let (bounds, kernel, ks) = coefficients(inSize: width, outSize: outW, filter: filter)
        var out = [UInt8](repeating: 0, count: outW * height * 3)
        let half = Int64(1) << Int64(precisionBits - 1)
        pixels.withUnsafeBufferPointer { src in
            out.withUnsafeMutableBufferPointer { dst in
                for y in 0 ..< height {
                    let row = y * width * 3
                    for xx in 0 ..< outW {
                        let (xmin, xmax) = bounds[xx]
                        var r = half, g = half, b = half
                        for x in 0 ..< xmax {
                            let c = Int64(kernel[xx * ks + x])
                            let p = row + (xmin + x) * 3
                            r += Int64(src[p]) * c
                            g += Int64(src[p + 1]) * c
                            b += Int64(src[p + 2]) * c
                        }
                        let o = (y * outW + xx) * 3
                        dst[o] = clip8(r)
                        dst[o + 1] = clip8(g)
                        dst[o + 2] = clip8(b)
                    }
                }
            }
        }
        return RGBImage(width: outW, height: height, pixels: out)
    }

    private func resampledVertically(to outH: Int, filter: ResampleFilter) -> RGBImage {
        let (bounds, kernel, ks) = coefficients(inSize: height, outSize: outH, filter: filter)
        var out = [UInt8](repeating: 0, count: width * outH * 3)
        let half = Int64(1) << Int64(precisionBits - 1)
        pixels.withUnsafeBufferPointer { src in
            out.withUnsafeMutableBufferPointer { dst in
                for yy in 0 ..< outH {
                    let (ymin, ymax) = bounds[yy]
                    for x in 0 ..< width {
                        var r = half, g = half, b = half
                        for y in 0 ..< ymax {
                            let c = Int64(kernel[yy * ks + y])
                            let p = ((ymin + y) * width + x) * 3
                            r += Int64(src[p]) * c
                            g += Int64(src[p + 1]) * c
                            b += Int64(src[p + 2]) * c
                        }
                        let o = (yy * width + x) * 3
                        dst[o] = clip8(r)
                        dst[o + 1] = clip8(g)
                        dst[o + 2] = clip8(b)
                    }
                }
            }
        }
        return RGBImage(width: width, height: outH, pixels: out)
    }

    /// Pixels [x, x + w) x [y, y + h).
    public func cropped(x: Int, y: Int, width w: Int, height h: Int) -> RGBImage {
        var out = [UInt8](repeating: 0, count: w * h * 3)
        for r in 0 ..< h {
            let s = ((y + r) * width + x) * 3
            out.replaceSubrange((r * w * 3) ..< ((r + 1) * w * 3), with: pixels[s ..< s + w * 3])
        }
        return RGBImage(width: w, height: h, pixels: out)
    }

    /// `paste` of `other` at (x, y) (no clipping: it must fit).
    public mutating func paste(_ other: RGBImage, x: Int, y: Int) {
        for r in 0 ..< other.height {
            let d = ((y + r) * width + x) * 3
            let s = r * other.width * 3
            pixels.replaceSubrange(d ..< d + other.width * 3, with: other.pixels[s ..< s + other.width * 3])
        }
    }
}

// MARK: - frames.py

/// `grid_size`: largest multiple-of-`factor` size under `maxPixels`, aspect ratio kept, never upscaled.
public func gridSize(width: Int, height: Int, maxPixels: Int, factor: Int = 32) -> (Int, Int) {
    let scale = min(1.0, (Double(maxPixels) / Double(width * height)).squareRoot())
    var w = max(factor, Int((Double(width) * scale / Double(factor)).rounded(.toNearestOrEven)) * factor)
    var h = max(factor, Int((Double(height) * scale / Double(factor)).rounded(.toNearestOrEven)) * factor)
    while w * h > maxPixels {
        if w >= h { w -= factor } else { h -= factor }
    }
    return (w, h)
}

extension RGBImage {
    /// `letterbox`: fit into a side x side black square, centred.
    public func letterboxed(side: Int) -> RGBImage {
        if width == side && height == side { return self }
        let s = Double(side) / Double(max(width, height))
        let w = max(1, Int((Double(width) * s).rounded(.toNearestOrEven)))
        let h = max(1, Int((Double(height) * s).rounded(.toNearestOrEven)))
        let im = resized(width: w, height: h, filter: .bicubic)
        var sq = RGBImage(width: side, height: side)
        sq.paste(im, x: (side - w) / 2, y: (side - h) / 2)
        return sq
    }

    /// `fit_long_side`: aspect ratio kept, long side = `side`.
    public func fittingLongSide(_ side: Int) -> RGBImage {
        if max(width, height) == side { return self }
        let s = Double(side) / Double(max(width, height))
        return resized(
            width: max(1, Int((Double(width) * s).rounded(.toNearestOrEven))),
            height: max(1, Int((Double(height) * s).rounded(.toNearestOrEven))), filter: .bicubic)
    }

    /// The store's frame size (`grid_size` under `maxPixels`), bicubic.
    public func resizedForStore(maxPixels: Int) -> RGBImage {
        let (w, h) = gridSize(width: width, height: height, maxPixels: maxPixels)
        return resized(width: w, height: h, filter: .bicubic)
    }

    /// JPEG round trip at `quality` (1-100), like the reference's `jpeg_roundtrip` (Pillow + libjpeg-turbo):
    /// libjpeg-turbo's lossy steps computed directly (JPEGRoundTrip.swift).
    public func jpegRoundTrip(quality: Int?) -> RGBImage {
        guard let quality else { return self }
        return LibJPEG.roundTrip(self, quality: quality)
    }

    /// JPEG round trip through ImageIO's encoder (not libjpeg's: pixels differ from the reference).
    public func imageIOJPEGRoundTrip(quality: Int?) -> RGBImage {
        guard let quality, let cg = cgImage() else { return self }
        let data = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(data, UTType.jpeg.identifier as CFString, 1, nil)
        else { return self }
        CGImageDestinationAddImage(
            dest, cg, [kCGImageDestinationLossyCompressionQuality: Double(quality) / 100.0] as CFDictionary)
        guard CGImageDestinationFinalize(dest),
            let src = CGImageSourceCreateWithData(data, nil),
            let back = CGImageSourceCreateImageAtIndex(src, 0, nil),
            let img = RGBImage(cgImage: back)
        else { return self }
        return img
    }
}

/// `format_time`: "hms" (00:06:57) or "s" (417s).
public func formatTime(_ t: Double, style: String = "hms") -> String {
    let s = Int(t.rounded(.toNearestOrEven))
    if style == "hms" {
        return String(format: "%02d:%02d:%02d", s / 3600, (s % 3600) / 60, s % 60)
    }
    return "\(s)s"
}

// MARK: - CoreGraphics

extension RGBImage {
    /// Draws any CGImage into 8-bit RGB (sRGB, alpha dropped).
    public init?(cgImage: CGImage) {
        let w = cgImage.width, h = cgImage.height
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
            let ctx = CGContext(
                data: &rgba, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4, space: space,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { return nil }
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h))
        var rgb = [UInt8](repeating: 0, count: w * h * 3)
        for i in 0 ..< w * h {
            rgb[i * 3] = rgba[i * 4]
            rgb[i * 3 + 1] = rgba[i * 4 + 1]
            rgb[i * 3 + 2] = rgba[i * 4 + 2]
        }
        self.init(width: w, height: h, pixels: rgb)
    }

    public func cgImage() -> CGImage? {
        var rgba = [UInt8](repeating: 255, count: width * height * 4)
        for i in 0 ..< width * height {
            rgba[i * 4] = pixels[i * 3]
            rgba[i * 4 + 1] = pixels[i * 3 + 1]
            rgba[i * 4 + 2] = pixels[i * 3 + 2]
        }
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
            let provider = CGDataProvider(data: Data(rgba) as CFData)
        else { return nil }
        return CGImage(
            width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
            space: space, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
    }
}
