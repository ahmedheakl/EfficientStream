// The JPEG round trip of the reference (`jpeg_roundtrip`: Pillow's `save(format="JPEG", quality=q)` then decode),
// computed directly: libjpeg-turbo's lossy steps, without the (lossless) entropy coding.
//
// Encoder (Pillow defaults): RGB -> YCbCr (jccolor.c), 2x2 chroma downsampling with alternating bias
// (jcsample.c h2v2_downsample), edges replicated (expand_right_edge / expand_bottom_edge), the accurate integer
// forward DCT (jfdctint.c), quantization with the standard tables scaled by quality (jcparam.c) using
// libjpeg-turbo's reciprocal method (jcdctmgr.c, 16-bit SIMD build).
// Decoder: dequantization and the accurate integer inverse DCT (jidctint.c) with the IDCT range-limit table
// (jdmaster.c), "fancy" triangle upsampling of the chroma (jdsample.c h2v2_fancy_upsample, edge rows/columns
// replicated as jdmainct.c does), YCbCr -> RGB (jdcolor.c).

import Foundation

enum LibJPEG {
    // MARK: tables

    static let stdLuminance: [Int] = [
        16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
        14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
        49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
    ]
    static let stdChrominance: [Int] = [
        17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99,
        47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    ]

    /// jpeg_set_quality(quality, force_baseline = TRUE)
    static func quantTable(_ base: [Int], quality: Int) -> [Int] {
        let q = min(max(quality, 1), 100)
        let scale = q < 50 ? 5000 / q : 200 - q * 2
        return base.map { min(max(($0 * scale + 50) / 100, 1), 255) }
    }

    /// compute_reciprocal for a 16-bit DCTELEM: (reciprocal, correction, shift r) with q = ((|x| + c) * f) >> r.
    static func reciprocal(_ divisor: Int) -> (Int, Int, Int) {
        if divisor == 1 { return (1, 0, 16) }
        let b = (Int.bitWidth - divisor.leadingZeroBitCount) - 1  // flss(divisor) - 1
        var r = 16 + b
        var fq = (1 << r) / divisor
        let fr = (1 << r) % divisor
        var c = divisor / 2
        if fr == 0 {
            fq >>= 1
            r -= 1
        } else if fr <= divisor / 2 {
            c += 1
        } else {
            fq += 1
        }
        return (fq, c, r)
    }

    /// IDCT_range_limit: out = table[v & 1023] (clamps v + 128 to [0, 255] for -512 <= v < 512).
    static let idctRangeLimit: [UInt8] = {
        var t = [UInt8](repeating: 0, count: 1024)
        for k in 0 ..< 128 { t[k] = UInt8(128 + k) }
        for k in 128 ..< 512 { t[k] = 255 }
        for k in 896 ..< 1024 { t[k] = UInt8(k - 896) }
        return t
    }()

    // MARK: DCT (jfdctint.c / jidctint.c, CONST_BITS 13, PASS1_BITS 2)

    static let constBits = 13, pass1Bits = 2
    static let f0_298631336 = 2446, f0_390180644 = 3196, f0_541196100 = 4433, f0_765366865 = 6270
    static let f0_899976223 = 7373, f1_175875602 = 9633, f1_501321110 = 12299, f1_847759065 = 15137
    static let f1_961570560 = 16069, f2_053119869 = 16819, f2_562915447 = 20995, f3_072711026 = 25172

    @inline(__always) static func descale(_ x: Int, _ n: Int) -> Int { (x + (1 << (n - 1))) >> n }

    static func forwardDCT(_ d: inout [Int]) {
        for row in 0 ..< 8 {
            let o = row * 8
            let tmp0 = d[o] + d[o + 7], tmp7 = d[o] - d[o + 7]
            let tmp1 = d[o + 1] + d[o + 6], tmp6 = d[o + 1] - d[o + 6]
            let tmp2 = d[o + 2] + d[o + 5], tmp5 = d[o + 2] - d[o + 5]
            let tmp3 = d[o + 3] + d[o + 4], tmp4 = d[o + 3] - d[o + 4]
            let tmp10 = tmp0 + tmp3, tmp13 = tmp0 - tmp3, tmp11 = tmp1 + tmp2, tmp12 = tmp1 - tmp2
            d[o] = (tmp10 + tmp11) << pass1Bits
            d[o + 4] = (tmp10 - tmp11) << pass1Bits
            let z1 = (tmp12 + tmp13) * f0_541196100
            d[o + 2] = descale(z1 + tmp13 * f0_765366865, constBits - pass1Bits)
            d[o + 6] = descale(z1 + tmp12 * -f1_847759065, constBits - pass1Bits)
            var (a4, a5, a6, a7) = (tmp4, tmp5, tmp6, tmp7)
            var z1o = a4 + a7, z2 = a5 + a6, z3 = a4 + a6, z4 = a5 + a7
            let z5 = (z3 + z4) * f1_175875602
            a4 *= f0_298631336; a5 *= f2_053119869; a6 *= f3_072711026; a7 *= f1_501321110
            z1o *= -f0_899976223; z2 *= -f2_562915447; z3 *= -f1_961570560; z4 *= -f0_390180644
            z3 += z5; z4 += z5
            d[o + 7] = descale(a4 + z1o + z3, constBits - pass1Bits)
            d[o + 5] = descale(a5 + z2 + z4, constBits - pass1Bits)
            d[o + 3] = descale(a6 + z2 + z3, constBits - pass1Bits)
            d[o + 1] = descale(a7 + z1o + z4, constBits - pass1Bits)
        }
        for col in 0 ..< 8 {
            func at(_ r: Int) -> Int { d[r * 8 + col] }
            let tmp0 = at(0) + at(7), tmp7 = at(0) - at(7)
            let tmp1 = at(1) + at(6), tmp6 = at(1) - at(6)
            let tmp2 = at(2) + at(5), tmp5 = at(2) - at(5)
            let tmp3 = at(3) + at(4), tmp4 = at(3) - at(4)
            let tmp10 = tmp0 + tmp3, tmp13 = tmp0 - tmp3, tmp11 = tmp1 + tmp2, tmp12 = tmp1 - tmp2
            d[col] = descale(tmp10 + tmp11, pass1Bits)
            d[32 + col] = descale(tmp10 - tmp11, pass1Bits)
            let z1 = (tmp12 + tmp13) * f0_541196100
            d[16 + col] = descale(z1 + tmp13 * f0_765366865, constBits + pass1Bits)
            d[48 + col] = descale(z1 + tmp12 * -f1_847759065, constBits + pass1Bits)
            var (a4, a5, a6, a7) = (tmp4, tmp5, tmp6, tmp7)
            var z1o = a4 + a7, z2 = a5 + a6, z3 = a4 + a6, z4 = a5 + a7
            let z5 = (z3 + z4) * f1_175875602
            a4 *= f0_298631336; a5 *= f2_053119869; a6 *= f3_072711026; a7 *= f1_501321110
            z1o *= -f0_899976223; z2 *= -f2_562915447; z3 *= -f1_961570560; z4 *= -f0_390180644
            z3 += z5; z4 += z5
            d[56 + col] = descale(a4 + z1o + z3, constBits + pass1Bits)
            d[40 + col] = descale(a5 + z2 + z4, constBits + pass1Bits)
            d[24 + col] = descale(a6 + z2 + z3, constBits + pass1Bits)
            d[8 + col] = descale(a7 + z1o + z4, constBits + pass1Bits)
        }
    }

    /// Dequantized coefficients -> 8x8 samples (written with `put(row, col, value)`).
    static func inverseDCT(_ coef: [Int], _ q: [Int], _ out: inout [UInt8], _ base: Int, _ stride: Int) {
        var ws = [Int](repeating: 0, count: 64)
        for col in 0 ..< 8 {
            func c(_ r: Int) -> Int { coef[r * 8 + col] * q[r * 8 + col] }
            // (the reference's all-zero-AC shortcut gives the same values as the full computation)
            var z2 = c(2), z3 = c(6)
            var z1 = (z2 + z3) * f0_541196100
            var tmp2 = z1 + z3 * -f1_847759065
            var tmp3 = z1 + z2 * f0_765366865
            z2 = c(0); z3 = c(4)
            var tmp0 = (z2 + z3) << constBits
            var tmp1 = (z2 - z3) << constBits
            let tmp10 = tmp0 + tmp3, tmp13 = tmp0 - tmp3, tmp11 = tmp1 + tmp2, tmp12 = tmp1 - tmp2
            tmp0 = c(7); tmp1 = c(5); tmp2 = c(3); tmp3 = c(1)
            z1 = tmp0 + tmp3; z2 = tmp1 + tmp2; z3 = tmp0 + tmp2
            var z4 = tmp1 + tmp3
            let z5 = (z3 + z4) * f1_175875602
            tmp0 *= f0_298631336; tmp1 *= f2_053119869; tmp2 *= f3_072711026; tmp3 *= f1_501321110
            z1 *= -f0_899976223; z2 *= -f2_562915447; z3 *= -f1_961570560; z4 *= -f0_390180644
            z3 += z5; z4 += z5
            tmp0 += z1 + z3; tmp1 += z2 + z4; tmp2 += z2 + z3; tmp3 += z1 + z4
            let n = constBits - pass1Bits
            ws[col] = descale(tmp10 + tmp3, n); ws[56 + col] = descale(tmp10 - tmp3, n)
            ws[8 + col] = descale(tmp11 + tmp2, n); ws[48 + col] = descale(tmp11 - tmp2, n)
            ws[16 + col] = descale(tmp12 + tmp1, n); ws[40 + col] = descale(tmp12 - tmp1, n)
            ws[24 + col] = descale(tmp13 + tmp0, n); ws[32 + col] = descale(tmp13 - tmp0, n)
        }
        let rl = idctRangeLimit
        for row in 0 ..< 8 {
            let w = row * 8
            var z2 = ws[w + 2], z3 = ws[w + 6]
            var z1 = (z2 + z3) * f0_541196100
            var tmp2 = z1 + z3 * -f1_847759065
            var tmp3 = z1 + z2 * f0_765366865
            var tmp0 = (ws[w] + ws[w + 4]) << constBits
            var tmp1 = (ws[w] - ws[w + 4]) << constBits
            let tmp10 = tmp0 + tmp3, tmp13 = tmp0 - tmp3, tmp11 = tmp1 + tmp2, tmp12 = tmp1 - tmp2
            tmp0 = ws[w + 7]; tmp1 = ws[w + 5]; tmp2 = ws[w + 3]; tmp3 = ws[w + 1]
            z1 = tmp0 + tmp3; z2 = tmp1 + tmp2; z3 = tmp0 + tmp2
            var z4 = tmp1 + tmp3
            let z5 = (z3 + z4) * f1_175875602
            tmp0 *= f0_298631336; tmp1 *= f2_053119869; tmp2 *= f3_072711026; tmp3 *= f1_501321110
            z1 *= -f0_899976223; z2 *= -f2_562915447; z3 *= -f1_961570560; z4 *= -f0_390180644
            z3 += z5; z4 += z5
            tmp0 += z1 + z3; tmp1 += z2 + z4; tmp2 += z2 + z3; tmp3 += z1 + z4
            let n = constBits + pass1Bits + 3
            let o = base + row * stride
            out[o] = rl[descale(tmp10 + tmp3, n) & 1023]
            out[o + 7] = rl[descale(tmp10 - tmp3, n) & 1023]
            out[o + 1] = rl[descale(tmp11 + tmp2, n) & 1023]
            out[o + 6] = rl[descale(tmp11 - tmp2, n) & 1023]
            out[o + 2] = rl[descale(tmp12 + tmp1, n) & 1023]
            out[o + 5] = rl[descale(tmp12 - tmp1, n) & 1023]
            out[o + 3] = rl[descale(tmp13 + tmp0, n) & 1023]
            out[o + 4] = rl[descale(tmp13 - tmp0, n) & 1023]
        }
    }

    /// Encode + decode one plane (already padded to whole blocks: `w`, `h` multiples of 8) with table `q`.
    static func lossyPlane(_ plane: [UInt8], w: Int, h: Int, q: [Int]) -> [UInt8] {
        let recips = q.map { reciprocal($0 << 3) }
        var out = [UInt8](repeating: 0, count: w * h)
        var block = [Int](repeating: 0, count: 64)
        var coef = [Int](repeating: 0, count: 64)
        for by in stride(from: 0, to: h, by: 8) {
            for bx in stride(from: 0, to: w, by: 8) {
                for r in 0 ..< 8 {
                    for c in 0 ..< 8 { block[r * 8 + c] = Int(plane[(by + r) * w + bx + c]) - 128 }
                }
                forwardDCT(&block)
                for i in 0 ..< 64 {
                    let (f, corr, sh) = recips[i]
                    var t = block[i]
                    let neg = t < 0
                    if neg { t = -t }
                    var v = Int(UInt32(truncatingIfNeeded: t + corr) &* UInt32(f)) >> sh
                    if neg { v = -v }
                    coef[i] = v
                }
                inverseDCT(coef, q, &out, by * w + bx, w)
            }
        }
        return out
    }

    // MARK: color (16-bit fixed point)

    static func fix(_ x: Double) -> Int { Int(x * 65536.0 + 0.5) }
    static let oneHalf = 1 << 15

    static func roundTrip(_ img: RGBImage, quality: Int) -> RGBImage {
        let (W, H) = (img.width, img.height)
        // encoder geometry (h2v2): luma blocks cover W x H; chroma covers ceil(W/2) x ceil(H/2)
        let yBW = (W + 7) / 8, yBH = (H + 7) / 8
        let cw = (W + 1) / 2, ch = (H + 1) / 2
        let cBW = (cw + 7) / 8, cBH = (ch + 7) / 8
        let yW = yBW * 8, yH = yBH * 8, cW = cBW * 8, cH = cBH * 8

        // RGB -> YCbCr at full resolution (jccolext.c), for the image columns; then edge replication
        let fyR = fix(0.29900), fyG = fix(0.58700), fyB = fix(0.11400)
        let fcbR = -fix(0.16874), fcbG = -fix(0.33126), fHalf = fix(0.50000)
        let fcrG = -fix(0.41869), fcrB = -fix(0.08131)
        let cbcrOffset = 128 << 16
        // full-res planes padded to the widths the downsamplers read: luma yW, chroma 2*cW; rows to even
        let fullW = max(yW, 2 * cW), fullH = max(yH, 2 * cH)
        var Y = [UInt8](repeating: 0, count: fullW * fullH)
        var Cb = Y, Cr = Y
        img.pixels.withUnsafeBufferPointer { px in
            for y in 0 ..< H {
                for x in 0 ..< W {
                    let p = (y * W + x) * 3
                    let r = Int(px[p]), g = Int(px[p + 1]), b = Int(px[p + 2])
                    let i = y * fullW + x
                    Y[i] = UInt8((fyR * r + fyG * g + fyB * b + oneHalf) >> 16)
                    Cb[i] = UInt8((fcbR * r + fcbG * g + fHalf * b + cbcrOffset + oneHalf - 1) >> 16)
                    Cr[i] = UInt8((fHalf * r + fcrG * g + fcrB * b + cbcrOffset + oneHalf - 1) >> 16)
                }
                for x in W ..< fullW {  // expand_right_edge
                    let i = y * fullW + x, s = y * fullW + W - 1
                    Y[i] = Y[s]; Cb[i] = Cb[s]; Cr[i] = Cr[s]
                }
            }
        }
        for y in H ..< fullH {  // expand_bottom_edge
            let s = (H - 1) * fullW, d = y * fullW
            for x in 0 ..< fullW { Y[d + x] = Y[s + x]; Cb[d + x] = Cb[s + x]; Cr[d + x] = Cr[s + x] }
        }

        // luma plane (fullsize_downsample) and chroma planes (h2v2_downsample, bias 1,2,1,2,...)
        var yPlane = [UInt8](repeating: 0, count: yW * yH)
        for y in 0 ..< yH { for x in 0 ..< yW { yPlane[y * yW + x] = Y[y * fullW + x] } }
        func down(_ src: [UInt8]) -> [UInt8] {
            var o = [UInt8](repeating: 0, count: cW * cH)
            let rows = (H + 1) / 2  // rows computed from image (and duplicated last) rows
            for oy in 0 ..< rows {
                var bias = 1
                let r0 = (2 * oy) * fullW, r1 = (2 * oy + 1) * fullW
                for ox in 0 ..< cW {
                    let s = Int(src[r0 + 2 * ox]) + Int(src[r0 + 2 * ox + 1]) + Int(src[r1 + 2 * ox])
                        + Int(src[r1 + 2 * ox + 1])
                    o[oy * cW + ox] = UInt8((s + bias) >> 2)
                    bias ^= 3
                }
            }
            for oy in rows ..< cH {  // pad the output to whole blocks with the last row
                for ox in 0 ..< cW { o[oy * cW + ox] = o[(rows - 1) * cW + ox] }
            }
            return o
        }
        let qY = quantTable(stdLuminance, quality: quality)
        let qC = quantTable(stdChrominance, quality: quality)
        let dY = lossyPlane(yPlane, w: yW, h: yH, q: qY)
        let dCb = lossyPlane(down(Cb), w: cW, h: cH, q: qC)
        let dCr = lossyPlane(down(Cr), w: cW, h: cH, q: qC)

        // fancy upsampling (h2v2), rows above/below and columns at the edges replicated; libjpeg-turbo uses it
        // only when the chroma is wider than 2 samples, and plain 2x2 replication (h2v2_upsample) otherwise
        func up(_ p: [UInt8]) -> [UInt8] {
            let ow = 2 * cw
            var o = [UInt8](repeating: 0, count: ow * 2 * ch)
            if cw <= 2 {
                for oy in 0 ..< 2 * ch { for ox in 0 ..< ow { o[oy * ow + ox] = p[(oy / 2) * cW + ox / 2] } }
                return o
            }
            for iy in 0 ..< ch {
                for v in 0 ..< 2 {
                    let near = iy, far = v == 0 ? max(iy - 1, 0) : min(iy + 1, ch - 1)
                    let oy = 2 * iy + v
                    func colsum(_ x: Int) -> Int { Int(p[near * cW + x]) * 3 + Int(p[far * cW + x]) }
                    var this = colsum(0), next = colsum(1), last = this
                    o[oy * ow] = UInt8((this * 4 + 8) >> 4)
                    o[oy * ow + 1] = UInt8((this * 3 + next + 7) >> 4)
                    last = this; this = next
                    for x in 1 ..< cw - 1 {
                        next = colsum(x + 1)
                        o[oy * ow + 2 * x] = UInt8((this * 3 + last + 8) >> 4)
                        o[oy * ow + 2 * x + 1] = UInt8((this * 3 + next + 7) >> 4)
                        last = this; this = next
                    }
                    o[oy * ow + 2 * (cw - 1)] = UInt8((this * 3 + last + 8) >> 4)
                    o[oy * ow + 2 * (cw - 1) + 1] = UInt8((this * 4 + 7) >> 4)
                }
            }
            return o
        }
        let uCb = up(dCb), uCr = up(dCr)

        // YCbCr -> RGB (jdcolext.c)
        var crR = [Int](repeating: 0, count: 256), cbB = crR, crG = crR, cbG = crR
        for i in 0 ..< 256 {
            let x = i - 128
            crR[i] = (fix(1.40200) * x + oneHalf) >> 16
            cbB[i] = (fix(1.77200) * x + oneHalf) >> 16
            crG[i] = -fix(0.71414) * x
            cbG[i] = -fix(0.34414) * x + oneHalf
        }
        @inline(__always) func clamp(_ v: Int) -> UInt8 { v < 0 ? 0 : (v > 255 ? 255 : UInt8(v)) }
        var out = [UInt8](repeating: 0, count: W * H * 3)
        let uw = 2 * cw
        for y in 0 ..< H {
            for x in 0 ..< W {
                let yy = Int(dY[y * yW + x]), cb = Int(uCb[y * uw + x]), cr = Int(uCr[y * uw + x])
                let o = (y * W + x) * 3
                out[o] = clamp(yy + crR[cr])
                out[o + 1] = clamp(yy + ((cbG[cb] + crG[cr]) >> 16))
                out[o + 2] = clamp(yy + cbB[cb])
            }
        }
        return RGBImage(width: W, height: H, pixels: out)
    }
}
