import AVFoundation
import UIKit

/// A video file played in real time as a frame source: AVPlayer for display, AVPlayerItemVideoOutput for frames.
/// Stream time is the playback position; seeking backwards restarts the stream.
@Observable
final class VideoFileSource: @unchecked Sendable {
    let player = AVPlayer()
    private(set) var url: URL?
    private(set) var duration: Double = 0
    private(set) var position: Double = 0
    private(set) var isPlaying = false

    /// (playback time, pixel buffer, rotation in degrees to make the frame upright)
    @ObservationIgnored var onFrame: (@Sendable (Double, CVPixelBuffer, Int) -> Void)?
    /// Playback jumped backwards: the stream must restart.
    @ObservationIgnored var onRestart: (() -> Void)?

    @ObservationIgnored private var output: AVPlayerItemVideoOutput?
    @ObservationIgnored private var link: CADisplayLink?
    @ObservationIgnored private var rotation = 0
    @ObservationIgnored private var lastTime = -1.0
    @ObservationIgnored private var endObserver: Any?

    func load(_ url: URL) async {
        stop()
        self.url = url
        let asset = AVURLAsset(url: url)
        duration = (try? await asset.load(.duration).seconds) ?? 0
        if let track = try? await asset.loadTracks(withMediaType: .video).first,
            let t = try? await track.load(.preferredTransform)
        {
            let angle = Int((atan2(t.b, t.a) * 180 / .pi).rounded())
            rotation = (angle + 360) % 360
        }
        let item = AVPlayerItem(asset: asset)
        let out = AVPlayerItemVideoOutput(pixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ])
        item.add(out)
        output = out
        player.replaceCurrentItem(with: item)
        lastTime = -1
        endObserver = NotificationCenter.default.addObserver(
            forName: AVPlayerItem.didPlayToEndTimeNotification, object: item, queue: .main
        ) { [weak self] _ in self?.isPlaying = false }
        let l = CADisplayLink(target: DisplayLinkTarget { [weak self] in self?.tick() }, selector: #selector(DisplayLinkTarget.fire))
        l.preferredFrameRateRange = CAFrameRateRange(minimum: 10, maximum: 30)
        l.add(to: .main, forMode: .common)
        link = l
    }

    func play() {
        player.play()
        isPlaying = true
    }

    func pause() {
        player.pause()
        isPlaying = false
    }

    func restart() {
        player.seek(to: .zero)
        lastTime = -1
        onRestart?()
        play()
    }

    func stop() {
        player.pause()
        isPlaying = false
        link?.invalidate()
        link = nil
        if let endObserver { NotificationCenter.default.removeObserver(endObserver) }
        endObserver = nil
    }

    private func tick() {
        guard let output else { return }
        let itemTime = output.itemTime(forHostTime: CACurrentMediaTime())
        position = player.currentTime().seconds.isFinite ? player.currentTime().seconds : 0
        guard output.hasNewPixelBuffer(forItemTime: itemTime),
            let pb = output.copyPixelBuffer(forItemTime: itemTime, itemTimeForDisplay: nil)
        else { return }
        let t = itemTime.seconds
        if t + 0.5 < lastTime { onRestart?() }
        lastTime = t
        onFrame?(t, pb, rotation)
    }
}

/// CADisplayLink retains its target; this breaks the cycle.
final class DisplayLinkTarget: NSObject {
    let action: () -> Void
    init(_ action: @escaping () -> Void) { self.action = action }
    @objc func fire() { action() }
}
