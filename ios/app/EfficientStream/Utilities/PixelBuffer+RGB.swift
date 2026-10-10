import CoreVideo
import EfficientStreamKit
import UIKit

extension RGBImage {
    /// A BGRA pixel buffer as 8-bit RGB, rotated clockwise by `rotation` degrees (0, 90, 180, 270).
    init(pixelBuffer pb: CVPixelBuffer, rotation: Int = 0) {
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        let stride = CVPixelBufferGetBytesPerRow(pb)
        let base = CVPixelBufferGetBaseAddress(pb)!.assumingMemoryBound(to: UInt8.self)
        let (ow, oh) = (rotation == 90 || rotation == 270) ? (h, w) : (w, h)
        var out = [UInt8](repeating: 0, count: ow * oh * 3)
        out.withUnsafeMutableBufferPointer { dst in
            for y in 0 ..< h {
                let row = base + y * stride
                for x in 0 ..< w {
                    let (ox, oy): (Int, Int)
                    switch rotation {
                    case 90: (ox, oy) = (h - 1 - y, x)
                    case 180: (ox, oy) = (w - 1 - x, h - 1 - y)
                    case 270: (ox, oy) = (y, w - 1 - x)
                    default: (ox, oy) = (x, y)
                    }
                    let s = row + x * 4
                    let d = (oy * ow + ox) * 3
                    dst[d] = s[2]
                    dst[d + 1] = s[1]
                    dst[d + 2] = s[0]
                }
            }
        }
        self.init(width: ow, height: oh, pixels: out)
    }

    var uiImage: UIImage? { cgImage().map { UIImage(cgImage: $0) } }
}
