import AVFoundation
import UIKit

/// The live camera as a frame source (adapted from Mobile-O's CameraManager): a native preview plus every
/// frame delivered, upright, as a BGRA pixel buffer with its stream time.
@Observable
final class CameraSource: NSObject, @unchecked Sendable {
    let session = AVCaptureSession()
    private(set) var isRunning = false
    private(set) var permissionDenied = false

    var useBackCamera = true

    /// Called on the capture queue for every frame: (seconds since the stream started, pixel buffer).
    @ObservationIgnored var onFrame: (@Sendable (Double, CVPixelBuffer) -> Void)?

    @ObservationIgnored private let sessionQueue = DispatchQueue(label: "camera.session")
    @ObservationIgnored private let frameQueue = DispatchQueue(label: "camera.frames", qos: .userInitiated)
    @ObservationIgnored private var startTime: CMTime?

    func start() {
        sessionQueue.async { [self] in
            switch AVCaptureDevice.authorizationStatus(for: .video) {
            case .authorized: run()
            case .notDetermined:
                AVCaptureDevice.requestAccess(for: .video) { ok in
                    if ok { self.sessionQueue.async { self.run() } } else { self.setDenied() }
                }
            default: setDenied()
            }
        }
    }

    func stop() {
        sessionQueue.async { [self] in
            session.stopRunning()
            DispatchQueue.main.async { self.isRunning = false }
        }
    }

    /// Switches cameras; the stream restarts (times start again at 0).
    func switchCamera() {
        useBackCamera.toggle()
        sessionQueue.async { [self] in
            session.stopRunning()
            run()
        }
    }

    private func setDenied() { DispatchQueue.main.async { self.permissionDenied = true } }

    private func run() {
        configure()
        startTime = nil
        session.startRunning()
        DispatchQueue.main.async { self.isRunning = true }
    }

    private func configure() {
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        session.inputs.forEach { session.removeInput($0) }
        session.outputs.forEach { session.removeOutput($0) }
        if session.canSetSessionPreset(.hd1920x1080) { session.sessionPreset = .hd1920x1080 }
        let position: AVCaptureDevice.Position = useBackCamera ? .back : .front
        guard
            let device = AVCaptureDevice.DiscoverySession(
                deviceTypes: [.builtInWideAngleCamera], mediaType: .video, position: position
            ).devices.first,
            let input = try? AVCaptureDeviceInput(device: device), session.canAddInput(input)
        else { return }
        session.addInput(input)
        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: frameQueue)
        guard session.canAddOutput(output) else { return }
        session.addOutput(output)
        if let c = output.connection(with: .video) {
            if c.isVideoRotationAngleSupported(90) { c.videoRotationAngle = 90 }  // portrait, upright
            if c.isVideoMirroringSupported { c.isVideoMirrored = position == .front }
        }
    }
}

extension CameraSource: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sb: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pb = sb.imageBuffer else { return }
        let pts = sb.presentationTimeStamp
        if startTime == nil { startTime = pts }
        onFrame?(max(0, (pts - startTime!).seconds), pb)
    }
}
