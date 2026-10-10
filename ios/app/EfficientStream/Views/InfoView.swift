import EfficientStreamKit
import SwiftUI

struct InfoView: View {
    let controller: StreamController
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                Section("Pipeline") {
                    row("Answering model", "MiniCPM-V-4.6 (bfloat16, MLX)")
                    row("Frame index", "MobileCLIP2-S0 (Core ML)")
                    row("Stored frames", "one every \(Int(1 / controller.session.config.gridFps)) s, at most \(controller.session.config.storeBudget)")
                    row("Shown per question", "\(controller.session.config.retrieveK) past moments + the current frame")
                    row("Last frame indexed in", String(format: "%.2f s", controller.lastStoreSeconds))
                }
                Section("How it works") {
                    Text("While the feed runs, a frame every 2 s is resized, embedded by MobileCLIP2-S0 and kept in a bounded store that always spans the whole stream. Its thumbnail also goes through the vision tower once. A question is embedded too; the 16 most similar stored frames (in time order) and the current frame at high resolution go to MiniCPM-V-4.6 in one pass. Multiple choice reads the option letter from that pass; open questions are answered word by word.")
                        .font(.footnote)
                }
                Section("Credits") {
                    Text("App derived from Mobile-O (CC BY-NC-SA 4.0). MiniCPM-V-4.6: Apache-2.0 (OpenBMB). MobileCLIP2-S0: Apple Machine Learning Research Model License (research use only).")
                        .font(.footnote)
                }
            }
            .navigationTitle("About")
            .toolbar { Button("Done") { dismiss() } }
        }
    }

    private func row(_ k: String, _ v: String) -> some View {
        HStack { Text(k); Spacer(); Text(v).foregroundStyle(.secondary).multilineTextAlignment(.trailing) }
            .font(.footnote)
    }
}
