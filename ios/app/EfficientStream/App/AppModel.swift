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
    private(set) var es: EfficientStream?
    /// The streaming Qwen3.5-0.8B (loaded instead of the pipeline when one of its modes is requested).
    private(set) var qwen: StreamQwen?
    private(set) var qwenController: QwenStreamController?

    enum Model: String {
        case efficientStream, qwen
    }

    /// The model the stream screen uses (one is loaded at a time).
    private(set) var model = Model(rawValue: UserDefaults.standard.string(forKey: "model") ?? "") ?? .qwen
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
        if QwenBenchModel.requested {
            state = .loading
            let start = Date()
            do {
                qwen = try await StreamQwen(directory: QwenBenchModel.directory)
                loadSeconds = Date().timeIntervalSince(start)
                print(String(format: "ESQWEN loaded in %.1f s", loadSeconds))
                state = .ready
            } catch {
                state = .failed(error.localizedDescription)
            }
            return
        }
        if model == .qwen && StreamQwen.modelsPresent(in: QwenBenchModel.directory) {
            state = .loading
            do {
                let q = try await StreamQwen(directory: QwenBenchModel.directory)
                StreamQwen.limitGPUCache(megabytes: 512)
                qwenController = QwenStreamController(qwen: q)
                state = .ready
            } catch {
                state = .failed(error.localizedDescription)
            }
            return
        }
        guard let directory = Self.installedModelsDirectory else {
            state = .needsModels
            return
        }
        state = .loading
        let start = Date()
        do {
            let es = try await EfficientStream(directory: directory, cacheDirectory: Self.cacheDirectory)
            self.es = es
            if BenchModel.requestedRuns == nil { controller = StreamController(es: es) }
            loadSeconds = Date().timeIntervalSince(start)
            state = .ready
        } catch {
            state = .failed(error.localizedDescription)
        }
    }

    /// Unloads the current model and loads the other one.
    func switchModel(to m: Model) async {
        guard m != model else { return }
        model = m
        UserDefaults.standard.set(m.rawValue, forKey: "model")
        controller?.stop()
        qwenController?.stop()
        controller = nil
        es = nil
        qwenController = nil
        StreamQwen.clearGPUCache()
        state = .needsModels
        await loadIfPossible()
    }

    func retry() async {
        state = .needsModels
        await loadIfPossible()
    }
}
