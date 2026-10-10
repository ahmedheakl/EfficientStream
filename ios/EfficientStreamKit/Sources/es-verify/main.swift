// es-verify: compares the Swift port with tensors dumped from the PyTorch package.
//
//   swift run -c release es-verify <exported models dir> <reference dir from tools/dump_reference.py>
//
// Per question: (1) vision features from the reference's own view pixels, (2) the Swift image processor's views
// and (3) token ids from the reference's input images, (4) the answer from the Swift prompt + features.

import EfficientStreamKit
import Foundation
import MLX

struct Case: Decodable {
    let tag: String
    let now: Double
    let question: String
    let options: [String]?
    let view_hw: [[Int]]
    let answer: String
    let probs: [Double]?
    let frame_times: [Double]
}

struct Meta: Decodable { let cases: [Case] }

func cosine(_ a: MLXArray, _ b: MLXArray) -> (min: Float, mean: Float) {
    let a = a.asType(.float32), b = b.asType(.float32)
    let c = (a * b).sum(axis: -1) / (sqrt((a * a).sum(axis: -1)) * sqrt((b * b).sum(axis: -1)))
    return (c.min().item(Float.self), c.mean().item(Float.self))
}

/// Reference view (3, 14, h*w*14) in the processor's patch-packed layout -> [h*14, w*14, 3].
func unpack(_ packed: MLXArray, h: Int, w: Int) -> MLXArray {
    let n = h * w
    return packed.reshaped(3, 14, n, 14).transposed(2, 1, 3, 0)  // n, py, px, c
        .reshaped(h, w, 14, 14, 3).transposed(0, 2, 1, 3, 4).reshaped(h * 14, w * 14, 3)
}

func rgb(_ a: MLXArray) -> RGBImage {
    let (h, w) = (a.dim(0), a.dim(1))
    return RGBImage(width: w, height: h, pixels: a.asType(.uint8).reshaped(-1).asArray(UInt8.self))
}

@main
struct Verify {
    static func main() async throws {
        let args = CommandLine.arguments
        if args.count >= 4 && args[1] == "index" {
            try checkIndex(models: URL(fileURLWithPath: args[2]), file: URL(fileURLWithPath: args[3]))
            return
        }
        guard args.count >= 3 else {
            print("usage: es-verify <models dir> <reference dir> | es-verify index <models dir> <index.safetensors>")
            exit(2)
        }
        let models = URL(fileURLWithPath: args[1]), ref = URL(fileURLWithPath: args[2])
        let t0 = Date()
        let answerer = try await Answerer(directory: models.appendingPathComponent("minicpm"))
        print(String(format: "loaded in %.1fs", Date().timeIntervalSince(t0)))
        let meta = try JSONDecoder().decode(Meta.self, from: Data(contentsOf: ref.appendingPathComponent("meta.json")))
        let config = try StreamConfig.load(models.appendingPathComponent("efficientstream_config.json"))
        var failures = 0

        for c in meta.cases {
            let d = try loadArrays(url: ref.appendingPathComponent(c.tag + ".safetensors"))
            let refIDs = d["ids"]!.asArray(Int32.self).map(Int.init)
            let refFeats = d["vision_feats"]!

            // (1) vision features of the reference's views
            var views: [ImageView] = []
            for (i, hw) in c.view_hw.enumerated() {
                let px = unpack(d["view\(i)"]!, h: hw[0], w: hw[1])
                views.append(ImageView(image: rgb(((px + 1) * 127.5).round()), gridH: hw[0], gridW: hw[1]))
            }
            let feats = answerer.features(of: views)
            let (cmin, cmean) = cosine(feats, refFeats)

            // (2) + (3) the Swift processor and prompt from the reference's input images
            var images: [EncodedImage] = []
            var parts: [PromptPart] = []
            var swiftViews: [ImageView] = []
            for (i, t) in c.frame_times.enumerated() {
                let im = rgb(d["image\(i)"]!)
                let p = MiniCPMImageProcessor.process(im, scaleResolution: config.thumbSide, maxSlices: 1)
                swiftViews += p.views
                images.append(answerer.encode(im, scaleResolution: config.thumbSide, maxSlices: 1))
                parts += [.text("frame time[\(formatTime(t, style: config.timestamps))]: "),
                          .image(placeholder: images.last!.placeholder), .text("\n")]
            }
            let latest = rgb(d["image\(c.frame_times.count)"]!)
            let pl = MiniCPMImageProcessor.process(
                latest, scaleResolution: config.thumbSide, maxSlices: config.latestMaxSlices)
            swiftViews += pl.views
            images.append(answerer.encode(latest, scaleResolution: config.thumbSide, maxSlices: config.latestMaxSlices))
            parts += [.text("frame time[\(formatTime(c.now, style: config.timestamps))] (now): "),
                      .image(placeholder: images.last!.placeholder), .text("\n"),
                      .text(PromptText.questionBlock(c.question, options: c.options))]
            var maxPixelDiff = 0
            let sameGrid = swiftViews.count == c.view_hw.count
                && zip(swiftViews, c.view_hw).allSatisfy { $0.gridH == $1[0] && $0.gridW == $1[1] }
            if sameGrid {
                for (sv, rv) in zip(swiftViews, views) {
                    for (x, y) in zip(sv.image.pixels, rv.image.pixels) { maxPixelDiff = max(maxPixelDiff, abs(Int(x) - Int(y))) }
                }
            }
            let ids = answerer.tokenize(answerer.spec.render(parts: parts)) + answerer.spec.answerOpenIds
            let sameIDs = ids == refIDs

            // (4) the answer
            let t1 = Date()
            let a = answerer.answer(parts: parts, images: images, numOptions: c.options?.count ?? 0)
            let secs = Date().timeIntervalSince(t1)
            let sameAnswer = a.text == c.answer
            if !(sameIDs && sameGrid) { failures += 1 }
            print("\(c.tag): vision cos min \(String(format: "%.4f", cmin)) mean \(String(format: "%.4f", cmean))"
                + " | views \(sameGrid ? "same grids" : "DIFFERENT GRIDS"), max pixel diff \(maxPixelDiff)"
                + " | ids \(sameIDs ? "identical (\(ids.count))" : "DIFFER (\(ids.count) vs \(refIDs.count))")"
                + " | answer \(sameAnswer ? "same" : "DIFFERENT") \(String(format: "%.2fs", secs))")
            if !sameIDs, let k = zip(ids, refIDs).enumerated().first(where: { $0.element.0 != $0.element.1 })?.offset {
                print("   first id difference at \(k): swift \(ids[max(0, k - 3) ..< min(ids.count, k + 4)])"
                    + " ref \(refIDs[max(0, k - 3) ..< min(refIDs.count, k + 4)])")
            }
            if let p = a.probs {
                print("   swift \(a.text) \(p.map { String(format: "%.4f", $0) })  ref \(c.answer) \((c.probs ?? []).map { String(format: "%.4f", $0) })")
            } else {
                print("   swift: \(a.text)\n   ref:   \(c.answer)")
            }
        }
        print(failures == 0 ? "OK: prompts and views match" : "\(failures) case(s) with prompt or view differences")
    }
}
