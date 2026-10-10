import SwiftUI

/// Progress of the on-phone test set (BenchModel).
struct BenchView: View {
    let bench: BenchModel

    var body: some View {
        NavigationStack {
            List {
                Section {
                    Text(bench.status).font(.headline)
                    if bench.total > 0 {
                        ProgressView(value: Double(bench.done), total: Double(bench.total))
                        Text("\(bench.done) / \(bench.total) questions").font(.footnote).monospacedDigit()
                    }
                    Text("Keep the app open; results go to Documents/bench/results.jsonl.")
                        .font(.footnote).foregroundStyle(.secondary)
                }
                Section("Latest") {
                    ForEach(bench.lines) { Text($0.text).font(.caption2.monospaced()) }
                }
            }
            .navigationTitle("Test set")
        }
    }
}
