import EfficientStreamKit
import PhotosUI
import SwiftUI
import UniformTypeIdentifiers

/// The main screen: the live feed (camera or a playing video) on top, answers below, and the question bar.
/// Questions can be asked at any moment while the feed runs; each answer is about "now".
struct StreamView: View {
    let controller: StreamController
    @State private var question = ""
    @State private var multipleChoice = false
    @State private var options = ["", "", "", ""]
    @State private var pickerItem: PhotosPickerItem?
    @State private var loadingVideo = false
    @State private var showInfo = false
    @FocusState private var focused: Bool
    @Environment(\.scenePhase) private var scenePhase

    private let suggestions = [
        "What is happening right now?", "What happened just before this?", "What changed since the start?",
        "Describe the scene.",
    ]

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                feed
                    .frame(maxWidth: .infinity)
                    .frame(height: 340)
                    .clipped()
                    .overlay(alignment: .topLeading) { status }
                    .overlay(alignment: .bottom) { if controller.source == .video { videoControls } }
                answers
                QuestionBar(
                    question: $question, multipleChoice: $multipleChoice, options: $options,
                    busy: controller.isAnswering, focused: $focused, suggestions: suggestions
                ) { send() }
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle("EfficientStream")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { toolbar }
            .onAppear { controller.start() }
            .onDisappear { controller.stop() }
            .onChange(of: pickerItem) { _, item in if let item { load(item) } }
            // iOS does not allow GPU work in the background: stop feeding frames when the app is not active
            .onChange(of: scenePhase) { _, phase in
                if phase == .active { controller.start() } else if phase == .background { controller.stop() }
            }
            .sheet(isPresented: $showInfo) { InfoView(controller: controller) }
        }
    }

    // MARK: feed

    @ViewBuilder private var feed: some View {
        switch controller.source {
        case .camera:
            if controller.camera.permissionDenied {
                ContentUnavailableView(
                    "Camera access is off", systemImage: "video.slash",
                    description: Text("Allow camera access in Settings, or play a video instead."))
            } else {
                CameraPreviewView(session: controller.camera.session)
            }
        case .video:
            ZStack {
                Color.black
                if controller.video.url != nil {
                    VideoPlayerView(player: controller.video.player)
                } else {
                    PhotosPicker(selection: $pickerItem, matching: .videos) {
                        Label("Choose a video", systemImage: "film").font(.headline)
                    }
                }
                if loadingVideo { ProgressView().tint(.white) }
            }
        }
    }

    private var status: some View {
        HStack(spacing: 8) {
            HStack(spacing: 5) {
                Circle().fill(controller.source == .camera ? .red : .orange).frame(width: 8, height: 8)
                Text(formatTime(controller.now)).monospacedDigit()
            }
            Text("\(controller.storedCount)/\(controller.session.config.storeBudget) stored").monospacedDigit()
        }
        .font(.caption.weight(.semibold))
        .foregroundStyle(.white)
        .padding(.horizontal, 10).padding(.vertical, 6)
        .background(.black.opacity(0.55), in: Capsule())
        .padding(10)
    }

    private var videoControls: some View {
        HStack(spacing: 20) {
            Button { controller.video.restart() } label: { Image(systemName: "backward.end.fill") }
            Button { controller.video.isPlaying ? controller.video.pause() : controller.video.play() } label: {
                Image(systemName: controller.video.isPlaying ? "pause.fill" : "play.fill")
            }
            PhotosPicker(selection: $pickerItem, matching: .videos) { Image(systemName: "film.stack") }
        }
        .font(.title3)
        .foregroundStyle(.white)
        .padding(.horizontal, 18).padding(.vertical, 8)
        .background(.black.opacity(0.55), in: Capsule())
        .padding(10)
        .disabled(controller.video.url == nil)
    }

    // MARK: answers

    private var answers: some View {
        ScrollView {
            LazyVStack(spacing: 12) {
                if controller.history.isEmpty {
                    VStack(spacing: 6) {
                        Image(systemName: "text.bubble").font(.largeTitle).foregroundStyle(.secondary)
                        Text("Ask about the stream at any moment.").font(.headline)
                        Text("Frames are indexed every 2 s while the feed runs; a question looks at the current frame and the 16 most relevant past moments.")
                            .font(.footnote).foregroundStyle(.secondary).multilineTextAlignment(.center)
                    }
                    .padding(.top, 30).padding(.horizontal, 24)
                }
                ForEach(controller.history) { AnswerCard(qa: $0) }
            }
            .padding(12)
        }
        .scrollDismissesKeyboard(.interactively)
    }

    // MARK: toolbar

    @ToolbarContentBuilder private var toolbar: some ToolbarContent {
        ToolbarItem(placement: .topBarLeading) {
            Picker("Source", selection: Binding(get: { controller.source }, set: { controller.use($0) })) {
                ForEach(StreamController.Source.allCases) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)
            .frame(width: 150)
        }
        ToolbarItemGroup(placement: .topBarTrailing) {
            if controller.source == .camera {
                Button { controller.camera.switchCamera(); controller.resetStream() } label: {
                    Image(systemName: "arrow.triangle.2.circlepath.camera")
                }
            }
            Menu {
                Button("Restart stream", systemImage: "arrow.counterclockwise") { controller.resetStream() }
                Button("Clear answers", systemImage: "trash") { controller.clearHistory() }
                Button("About", systemImage: "info.circle") { showInfo = true }
            } label: { Image(systemName: "ellipsis.circle") }
        }
    }

    // MARK: actions

    private func send() {
        let opts: [String]? = multipleChoice
            ? options.enumerated().compactMap { i, o in
                let t = o.trimmingCharacters(in: .whitespaces)
                return t.isEmpty ? nil : "\(Character(UnicodeScalar(65 + i)!)). \(t)"
            } : nil
        controller.ask(question, options: (opts?.count ?? 0) >= 2 ? opts : nil)
        question = ""
        focused = false
    }

    private func load(_ item: PhotosPickerItem) {
        loadingVideo = true
        Task {
            defer { loadingVideo = false }
            if let movie = try? await item.loadTransferable(type: MovieFile.self) {
                await controller.openVideo(movie.url)
            }
        }
    }
}

/// A picked video, copied into the app's temporary folder.
struct MovieFile: Transferable {
    let url: URL
    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { SentTransferredFile($0.url) } importing: { received in
            let dest = FileManager.default.temporaryDirectory.appendingPathComponent(
                UUID().uuidString + "." + received.file.pathExtension)
            try FileManager.default.copyItem(at: received.file, to: dest)
            return MovieFile(url: dest)
        }
    }
}
