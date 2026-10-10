import SwiftUI

/// Shown until the models are in Documents/models.
struct ModelSetupView: View {
    let app: AppModel
    @State private var downloader = HubDownloader()
    @AppStorage("hubRepo") private var repo = "ahmedheakl/efficientstream"
    @AppStorage("hubFolder") private var folder = "ios"
    @State private var token = ""
    @State private var error: String?

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Label("EfficientStream needs its models on the device (about 2.9 GB).", systemImage: "shippingbox")
                    if case .failed(let message) = app.state {
                        Text(message).foregroundStyle(.red).font(.footnote)
                    }
                }
                Section("From your Mac (fastest)") {
                    Text("Connect the iPhone, then run in the repository:")
                    Text("ios/tools/install_models.sh").font(.system(.footnote, design: .monospaced))
                        .textSelection(.enabled)
                    Text("It copies ios/models into this app's Documents folder. The Files app works too: copy the `models` folder into On My iPhone › EfficientStream.")
                        .font(.footnote).foregroundStyle(.secondary)
                    Button("Check again") { Task { await app.retry() } }
                }
                Section("From Hugging Face") {
                    TextField("Repository", text: $repo).textInputAutocapitalization(.never).autocorrectionDisabled()
                    TextField("Folder", text: $folder).textInputAutocapitalization(.never).autocorrectionDisabled()
                    SecureField("Access token (private repos)", text: $token)
                    if downloader.isRunning {
                        ProgressView(value: downloader.progress) { Text(downloader.status).font(.footnote) }
                    } else {
                        Button("Download") { download() }.disabled(repo.isEmpty)
                    }
                    if let error { Text(error).foregroundStyle(.red).font(.footnote) }
                }
            }
            .navigationTitle("EfficientStream")
        }
    }

    private func download() {
        error = nil
        Task {
            do {
                try await downloader.download(repo: repo, folder: folder, token: token, to: AppModel.modelsDirectory)
                await app.retry()
            } catch {
                self.error = error.localizedDescription
            }
        }
    }
}
