// es-verify jpeg <jpeg.safetensors>: the Swift JPEG round trip against Pillow's (tools/dump_jpeg_reference.py).

import EfficientStreamKit
import Foundation
import MLX

func checkJPEG(file: URL, quality: Int = 88) throws {
    let d = try loadArrays(url: file)
    var i = 0
    var allExact = true
    while let input = d["in\(i)"], let ref = d["out\(i)"] {
        let img = rgb(input)
        let got = img.jpegRoundTrip(quality: quality)
        let r = ref.asType(.uint8).reshaped(-1).asArray(UInt8.self)
        var maxDiff = 0, differing = 0
        for (x, y) in zip(got.pixels, r) {
            let dd = abs(Int(x) - Int(y))
            if dd > 0 { differing += 1 }
            maxDiff = max(maxDiff, dd)
        }
        allExact = allExact && maxDiff == 0
        print("\(img.width)x\(img.height): " + (maxDiff == 0 ? "bit-exact"
            : "max diff \(maxDiff), \(differing) of \(r.count) values differ"))
        i += 1
    }
    print(allExact ? "OK: all \(i) round trips bit-exact" : "round trips differ")
}
