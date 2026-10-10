import EfficientStreamKit
import Foundation

/// Finds and loads the models. They live in Documents/models (copied from the Mac with
/// ios/tools/install_models.sh, through the Files app, or downloaded from Hugging Face in the app).
@Observable
@MainActor
final class AppModel {
    enum State: Equatable {
        case needsModels
        case loading
        case ready
        case failed(String)
    }

    private(set) var state: State = .needsModels
    private(set) var controller: StreamController?
    private(set) var loadSeconds: Double = 0

    static var documents: URL { FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0] }

    /// Where downloads go.
    static var modelsDirectory: URL { documents.appendingPathComponent("models", isDirectory: true) }

    /// The folder that holds the models: Documents/models, or one level deeper / shallower if a copy nested it.
    static var installedModelsDirectory: URL? {
        [modelsDirectory, modelsDirectory.appendingPathComponent("models", isDirectory: true), documents]
            .first { EfficientStream.modelsPresent(in: $0) }
    }

    static var cacheDirectory: URL {
        FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("coreml", isDirectory: true)
    }

    var modelsPresent: Bool { Self.installedModelsDirectory != nil }

    func loadIfPossible() async {
        guard state != .loading, state != .ready else { return }
        guard let directory = Self.installedModelsDirectory else {
            state = .needsModels
            return
        }
        state = .loading
        let start = Date()
        do {
            let es = try await EfficientStream(directory: directory, cacheDirectory: Self.cacheDirectory)
            controller = StreamController(es: es)
            loadSeconds = Date().timeIntervalSince(start)
            state = .ready
        } catch {
            state = .failed(error.localizedDescription)
        }
    }

    func retry() async {
        state = .needsModels
        await loadIfPossible()
    }
}
