import SwiftUI

/// App entry point. Gates the stream screen behind the models being on the device (as Mobile-O gates on its
/// model download).
@main
struct EfficientStreamApp: App {
    @State private var app = AppModel()
    @State private var bench = BenchModel()

    var body: some Scene {
        WindowGroup {
            Group {
                switch app.state {
                case .ready:
                    if let runs = BenchModel.requestedRuns, let es = app.es {
                        BenchView(bench: bench).task { await bench.run(es: es, runs: runs) }
                    } else if let controller = app.controller {
                        StreamView(controller: controller)
                    }
                case .loading:
                    LoadingView()
                case .needsModels, .failed:
                    ModelSetupView(app: app)
                }
            }
            .task { await app.loadIfPossible() }
        }
    }
}

private struct LoadingView: View {
    var body: some View {
        VStack(spacing: 16) {
            ProgressView().controlSize(.large)
            Text("Loading MiniCPM-V-4.6 and MobileCLIP2-S0…").font(.headline)
            Text("The first launch also compiles the Core ML encoder.").font(.footnote).foregroundStyle(.secondary)
        }
        .padding()
    }
}
