// Records the screen of the iPhone connected over USB, video only (capturing its audio makes iOS ask "Are you
// connecting a pair of headphones?" over the app). Each frame is stamped with its arrival time on the Mac, so
// movie time = wall time - the "first frame wall" time printed at the start; a stall shows as a held frame.
//
//   swiftc -O record_iphone.swift -o record_iphone
//   ./record_iphone take.mov <seconds | a path: stops when that file appears> > rec.log
import AVFoundation
import CoreMediaIO

setbuf(stdout, nil)
// let iOS devices appear as capture devices (as QuickTime does)
var prop = CMIOObjectPropertyAddress(
    mSelector: CMIOObjectPropertySelector(kCMIOHardwarePropertyAllowScreenCaptureDevices),
    mScope: CMIOObjectPropertyScope(kCMIOObjectPropertyScopeGlobal),
    mElement: CMIOObjectPropertyElement(kCMIOObjectPropertyElementMain))
var allow: UInt32 = 1
CMIOObjectSetPropertyData(
    CMIOObjectID(kCMIOObjectSystemObject), &prop, 0, nil, UInt32(MemoryLayout<UInt32>.size), &allow)

let out = URL(fileURLWithPath: CommandLine.arguments[1])
let until = CommandLine.arguments[2]
var device: AVCaptureDevice?
for _ in 0 ..< 30 {
    device = AVCaptureDevice.DiscoverySession(
        deviceTypes: [.external], mediaType: .muxed, position: .unspecified
    ).devices.first
    if device != nil { break }
    RunLoop.main.run(until: Date().addingTimeInterval(1))
}
guard let device else {
    print("no iPhone found (connected over USB, unlocked, trusted?)")
    exit(1)
}
let session = AVCaptureSession()
let input = try AVCaptureDeviceInput(device: device)
session.addInputWithNoConnections(input)
let output = AVCaptureVideoDataOutput()
output.alwaysDiscardsLateVideoFrames = false
output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange]
session.addOutputWithNoConnections(output)
guard let videoPort = input.ports.first(where: { $0.mediaType == .video }) else {
    print("no video port")
    exit(1)
}
session.addConnection(AVCaptureConnection(inputPorts: [videoPort], output: output))

/// HEVC (the H.264 encoder does not take the screen's 2622-pixel height).
final class Writer: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let url: URL
    private var writer: AVAssetWriter?
    private var input: AVAssetWriterInput?
    private var adaptor: AVAssetWriterInputPixelBufferAdaptor?
    private var t0 = 0.0
    private var lastT = -1.0
    private var frames = 0, dropped = 0
    private var maxGap = 0.0
    private var stopping = false
    private let lock = NSLock()

    init(url: URL) { self.url = url }

    func captureOutput(_ o: AVCaptureOutput, didOutput sb: CMSampleBuffer, from c: AVCaptureConnection) {
        let now = CACurrentMediaTime()
        let wall = Date().timeIntervalSince1970
        lock.lock()
        defer { lock.unlock() }
        guard !stopping, let pb = CMSampleBufferGetImageBuffer(sb) else { return }
        if writer == nil {
            let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
            guard let wr = try? AVAssetWriter(outputURL: url, fileType: .mov) else { return }
            let inp = AVAssetWriterInput(mediaType: .video, outputSettings: [
                AVVideoCodecKey: AVVideoCodecType.hevc, AVVideoWidthKey: w, AVVideoHeightKey: h,
            ])
            inp.expectsMediaDataInRealTime = true
            adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: inp, sourcePixelBufferAttributes: nil)
            wr.add(inp)
            wr.startWriting()
            wr.startSession(atSourceTime: .zero)
            writer = wr
            input = inp
            t0 = now
            print("first frame wall \(String(format: "%.3f", wall)) size \(w)x\(h)")
        }
        // frames can arrive in bursts: keep the timestamps strictly increasing
        var t = now - t0
        if lastT >= 0 {
            t = max(t, lastT + 0.002)
            maxGap = max(maxGap, t - lastT)
        }
        lastT = t
        if input!.isReadyForMoreMediaData,
            adaptor!.append(pb, withPresentationTime: CMTime(seconds: t, preferredTimescale: 1_000_000))
        {
            frames += 1
        } else {
            dropped += 1
        }
    }

    func finish() {
        lock.withLock { stopping = true }
        guard let writer, let input else {
            print("no frames")
            return
        }
        input.markAsFinished()
        writer.endSession(atSourceTime: CMTime(seconds: lastT, preferredTimescale: 1_000_000))
        let done = DispatchSemaphore(value: 0)
        writer.finishWriting { done.signal() }
        done.wait()
        print(String(format: "finished: %d frames, %d dropped, longest gap %.2f s, %@", frames, dropped, maxGap,
                     writer.error.map { "\($0)" } ?? "ok"))
    }
}

let writer = Writer(url: out)
try? FileManager.default.removeItem(at: out)
output.setSampleBufferDelegate(writer, queue: DispatchQueue(label: "frames"))
session.startRunning()
let deadline = Double(until).map { Date().addingTimeInterval($0) }
while true {
    RunLoop.main.run(until: Date().addingTimeInterval(0.2))
    if let deadline, Date() > deadline { break }
    if deadline == nil, FileManager.default.fileExists(atPath: until) { break }
}
session.stopRunning()
writer.finish()
