import EfficientStreamKit
import PhotosUI
import SwiftUI

/// The streaming Qwen3.5-0.8B: the live feed (camera or a playing video); the proactive mode (the eye button:
/// "Watch for…", the model says by itself when something happens); and questions at any moment, about the last
/// seconds and, once the stream is longer, the earlier moments most similar to the question (memory).
struct QwenStreamView: View {
    let controller: QwenStreamController
    var switchModel: () -> Void = {}
    @State private var question = ""
    @State private var multipleChoice = false
    @State private var options = ["", "", "", ""]
    @State private var pickerItem: PhotosPickerItem?
    @State private var loadingVideo = false
    @State private var editingWatch = false
    @State private var watchText = ""
    @FocusState private var focused: Bool
    @FocusState private var watchFocused: Bool
    @Environment(\.scenePhase) private var scenePhase

    private let suggestions = ["What is happening right now?", "What just happened?", "What is the person holding?"]
    private let watchSuggestions = ["Tell me when someone enters the room.", "Tell me when a person picks something up.",
                                    "Tell me when a car stops in front of us."]

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                feed
                    .frame(maxWidth: .infinity)
                    .frame(height: 340)
                    .clipped()
                    .overlay(alignment: .topLeading) { status }
                    .overlay(alignment: .bottom) { if controller.source == .video { videoControls } }
                if controller.watchPrompt != nil { watchBanner }
                if controller.captions { memoryStrip }
                answers
                if editingWatch { watchEditor }
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
            .task {
                if let script = QwenDemoModel.script {
                    await QwenDemoModel.run(controller, script: script)
                } else {
                    await controller.runLaunchChecks()
                }
            }
            .onChange(of: pickerItem) { _, item in if let item { load(item) } }
            .onChange(of: scenePhase) { _, phase in
                if phase == .active { controller.start() } else if phase == .background { controller.stop() }
            }
            .onChange(of: controller.demoWatchTyping) { _, t in
                if let t {
                    editingWatch = true
                    watchText = t
                    watchFocused = true
                }
            }
            .onChange(of: controller.demoWatchStarts) { _, _ in startWatch() }
            .onChange(of: controller.demoTyping) { _, t in
                if let t { question = t; focused = true }
            }
            .onChange(of: controller.demoSends) { _, _ in send() }
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
            if controller.captions {
                Text("\(controller.fps) fps").monospacedDigit()
            } else {
                Text("\(controller.framesInWindow)/\(controller.recent) s").monospacedDigit()
            }
            if controller.captions { Text("memory \(controller.memoryEvents.count)").monospacedDigit() }
            Text(String(format: "frame %.0f ms", (controller.lastFrame.prep + controller.lastFrame.vision) * 1000))
                .monospacedDigit()
            if controller.memoryMB > 0 { Text(String(format: "%.1f GB", controller.memoryMB / 1024)).monospacedDigit() }
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

    private var watchBanner: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Image(systemName: "eye").foregroundStyle(.tint)
                Text(controller.watchPrompt ?? "").font(.footnote.weight(.semibold)).lineLimit(2)
                Spacer()
                Button("Stop") { controller.watch(nil) }.font(.footnote)
            }
            if let u = controller.lastUpdate {
                Text(String(format: "%@ · second %d · update %.2f s (vision %.2f, prefill %d tok %.2f, answer %.2f)%@",
                            stateName(u.answer), u.second, u.total, u.vision, u.prefillTokens, u.prefill, u.decode,
                            controller.skippedSeconds > 0 ? " · \(controller.skippedSeconds) s skipped" : ""))
                    .font(.caption2.monospacedDigit()).foregroundStyle(.secondary)
            } else {
                Text("Waiting for the first second…").font(.caption2).foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 12).padding(.vertical, 8)
        .background(Color(.secondarySystemGroupedBackground))
    }

    /// The proactive mode's prompt: what to watch for.
    private var watchEditor: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Watch for something: the model stays silent and speaks up when it happens.")
                .font(.caption).foregroundStyle(.secondary)
            ScrollView(.horizontal, showsIndicators: false) {
                HStack {
                    ForEach(watchSuggestions, id: \.self) { s in
                        Button(s) { watchText = s }
                            .font(.caption).buttonStyle(.bordered).buttonBorderShape(.capsule)
                    }
                }
            }
            HStack(spacing: 10) {
                Image(systemName: "eye").font(.title3).foregroundStyle(.tint)
                TextField("Tell me when…", text: $watchText, axis: .vertical)
                    .lineLimit(1 ... 3)
                    .focused($watchFocused)
                    .padding(.horizontal, 14).padding(.vertical, 10)
                    .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 20))
                    .onSubmit { startWatch() }
                Button("Start") { startWatch() }
                    .buttonStyle(.borderedProminent)
                    .disabled(watchText.trimmingCharacters(in: .whitespaces).isEmpty)
                Button { editingWatch = false; watchFocused = false } label: { Image(systemName: "xmark") }
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 12).padding(.vertical, 8)
        .background(.bar)
    }

    private func startWatch() {
        let p = watchText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !p.isEmpty else { return }
        controller.watch(p)
        controller.demoWatchTyping = nil
        editingWatch = false
        watchFocused = false
    }

    private func stateName(_ answer: String) -> String {
        if answer.hasPrefix("</Response>") { return "Response" }
        if answer.hasPrefix("</Standby>") { return "Standby" }
        if answer.hasPrefix("</Silence>") { return "Silence" }
        return answer.isEmpty ? "-" : "Other"
    }

    /// The caption memory as it is written: the latest captions, newest first.
    private var memoryStrip: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack {
                Label("Memory", systemImage: "text.book.closed").font(.caption.weight(.semibold)).foregroundStyle(.tint)
                Spacer()
                if let u = controller.lastCaption {
                    Text(String(format: "second %d · %.2f s", u.second, u.total)).font(.caption2.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            }
            if controller.memoryEvents.isEmpty {
                Text("Watching… no captions yet").font(.caption).foregroundStyle(.secondary)
            }
            ForEach(Array(controller.memoryEvents.suffix(3).reversed().enumerated()), id: \.offset) { _, e in
                Text("\(e.kind == .summary ? "Summary" : "Seen") \(formatTime(e.start))  \(e.text)")
                    .font(.caption).lineLimit(2)
                    .foregroundStyle(e.kind == .summary ? .primary : .secondary)
            }
        }
        .padding(.horizontal, 12).padding(.vertical, 6)
        .background(Color(.secondarySystemGroupedBackground))
    }

    // MARK: answers

    private var answers: some View {
        ScrollView {
            LazyVStack(spacing: 12) {
                if controller.history.isEmpty && controller.events.isEmpty {
                    VStack(spacing: 6) {
                        Image(systemName: "text.bubble").font(.largeTitle).foregroundStyle(.secondary)
                        Text("Ask anything, at any moment.").font(.headline)
                        Text("While the video plays the model keeps writing captions of what it sees (its memory). A question reads them and looks at the last 16 seconds. Tap the eye to let the model watch for something and speak up by itself.")
                            .font(.footnote).foregroundStyle(.secondary).multilineTextAlignment(.center)
                    }
                    .padding(.top, 30).padding(.horizontal, 24)
                }
                ForEach(controller.events) { e in
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Label("Response", systemImage: "bell.fill").font(.caption.weight(.semibold))
                                .foregroundStyle(.tint)
                            Spacer()
                            Text("at \(formatTime(Double(e.second)))").font(.caption).foregroundStyle(.secondary)
                        }
                        Text(e.text.isEmpty ? "(no text)" : e.text)
                        Text(String(format: "update %.2f s", e.seconds)).font(.caption2.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                    .padding(12)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color(.systemBackground), in: RoundedRectangle(cornerRadius: 14))
                }
                ForEach(controller.history) { QwenAnswerCard(qa: $0) }
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
            Button {
                if controller.watchPrompt == nil {
                    editingWatch.toggle()
                    watchFocused = editingWatch
                } else {
                    controller.watch(nil)
                }
            } label: {
                Image(systemName: controller.watchPrompt == nil ? "eye" : "eye.fill")
            }
            .accessibilityLabel(controller.watchPrompt == nil ? "Watch for…" : "Stop watching")
            Menu {
                Button(controller.watchPrompt == nil ? "Watch for…" : "Stop watching", systemImage: "eye") {
                    if controller.watchPrompt == nil {
                        editingWatch = true
                        watchFocused = true
                    } else {
                        controller.watch(nil)
                    }
                }
                Button("Restart stream", systemImage: "arrow.counterclockwise") { controller.resetStream() }
                Button("Clear answers", systemImage: "trash") { controller.clearHistory() }
                Button("Use EfficientStream (MiniCPM-V-4.6)", systemImage: "arrow.triangle.swap") { switchModel() }
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

/// A question to the streaming Qwen3.5-0.8B and its answer, with where the time went.
struct QwenAnswerCard: View {
    let qa: QwenQA

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline) {
                Text(qa.question).font(.subheadline.weight(.semibold))
                Spacer()
                Text("at \(formatTime(qa.askedAt))").font(.caption).foregroundStyle(.secondary).monospacedDigit()
            }
            if let options = qa.options {
                ForEach(Array(options.enumerated()), id: \.offset) { i, o in
                    HStack {
                        Text(o).font(.footnote).fontWeight(String(o.prefix(1)) == qa.answer ? .bold : .regular)
                        Spacer()
                        if let p = qa.probs, i < p.count {
                            Text(String(format: "%.0f%%", p[i] * 100)).font(.caption2).foregroundStyle(.secondary)
                        }
                    }
                }
            } else if qa.answer.isEmpty && qa.isAnswering {
                HStack(spacing: 6) { ProgressView(); Text("Looking at the stream…").font(.footnote) }
                    .foregroundStyle(.secondary)
            } else {
                Text(qa.answer)
            }
            if let c = qa.context {
                Text("On the stream's KV cache: \(c.from)–\(c.through + 1) s, \(qa.frames) frames"
                     + (qa.memoryEvents > 0 ? " + \(qa.memoryEvents) earlier captions" : ""))
                    .font(.caption2).foregroundStyle(.tint)
            }
            if let r = qa.result {
                let decodeRate = r.decodeSeconds > 0 && r.generatedTokens > 1
                    ? Double(r.generatedTokens - 1) / r.decodeSeconds : 0
                Text(String(format: "%.1f tok/s · %d tokens · first words %.2f s · total %.2f s · %d frames · %d prompt tokens",
                            decodeRate, r.generatedTokens, r.tokenizeSeconds + r.prefillSeconds, qa.seconds, qa.frames,
                            r.promptTokens))
                    .font(.caption2.monospacedDigit()).foregroundStyle(.secondary)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.systemBackground), in: RoundedRectangle(cornerRadius: 14))
    }
}
