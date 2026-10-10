// es-verify bench <models dir>: timings of the question-time pieces.

import EfficientStreamKit
import Foundation
import MLX

func bench(models: URL) async throws {
    let a = try await Answerer(directory: models.appendingPathComponent("minicpm"))
    func time(_ name: String, _ n: Int = 3, _ f: () -> Void) {
        f()  // warm-up
        let t0 = Date()
        for _ in 0 ..< n { f() }
        print(String(format: "%@: %.3f s", name, Date().timeIntervalSince(t0) / Double(n)))
    }
    var img = RGBImage(width: 504, height: 896)
    for i in 0 ..< img.pixels.count { img.pixels[i] = UInt8((i * 7 + i / 3) % 256) }
    let thumb = RGBImage(width: 448, height: 448, fill: 90)
    time("latest frame 504x896 -> views + vision (3 views)") { _ = a.encode(img, scaleResolution: 448, maxSlices: 9) }
    time("thumbnail 448 -> vision (1 view)") { _ = a.encode(thumb, scaleResolution: 448, maxSlices: 1) }
    time("image processing only (latest)", 3) { _ = MiniCPMImageProcessor.process(img, scaleResolution: 448, maxSlices: 9) }
    time("JPEG round trip 504x896") { _ = img.jpegRoundTrip(quality: 88) }
    time("store resize 1920x1080") { _ = RGBImage(width: 1920, height: 1080, fill: 3).resizedForStore(maxPixels: 262144) }
    for n in [256, 1024, 1714] {
        let emb = MLXRandom.normal([1, n, 1024]).asType(.bfloat16) * 0.02
        time("prefill \(n) tokens (language model)") { a.benchPrefill(emb) }
    }
}
