import EfficientStreamKit
import SwiftUI

/// A question, its answer (streamed as it is generated), and the moments of the stream the model was shown.
struct AnswerCard: View {
    let qa: QA

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline) {
                Text(qa.question).font(.subheadline.weight(.semibold))
                Spacer()
                Text("at \(formatTime(qa.askedAt))").font(.caption).foregroundStyle(.secondary).monospacedDigit()
            }
            if let options = qa.options {
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(Array(options.enumerated()), id: \.offset) { i, o in
                        let letter = String(o.prefix(1))
                        HStack {
                            Text(o).font(.footnote)
                                .fontWeight(letter == qa.answer ? .bold : .regular)
                                .foregroundStyle(letter == qa.answer ? Color.accentColor : .primary)
                            Spacer()
                            if let p = qa.probs, i < p.count {
                                Text(String(format: "%.0f%%", p[i] * 100)).font(.caption2).foregroundStyle(.secondary)
                            }
                        }
                    }
                }
            }
            if let error = qa.error {
                Text(error).foregroundStyle(.red).font(.footnote)
            } else if qa.options == nil {
                if qa.answer.isEmpty && qa.isAnswering {
                    HStack(spacing: 6) { ProgressView(); Text("Looking at the stream…").font(.footnote) }
                        .foregroundStyle(.secondary)
                } else {
                    Text(qa.answer).font(.body)
                }
            } else if qa.isAnswering {
                ProgressView()
            }
            if let r = qa.result {
                Text(meta(r)).font(.caption2.monospacedDigit()).foregroundStyle(.secondary)
                DisclosureGroup("Where the time went") { TimingTable(t: r.timings) }
                    .font(.caption2).foregroundStyle(.secondary)
            }
            if !qa.thumbnails.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 6) {
                        ForEach(Array(qa.thumbnails.enumerated()), id: \.offset) { _, t in
                            VStack(spacing: 2) {
                                Image(uiImage: t.image).resizable().scaledToFill()
                                    .frame(width: 64, height: 40).clipped()
                                    .clipShape(RoundedRectangle(cornerRadius: 4))
                                Text(formatTime(t.time)).font(.system(size: 9)).monospacedDigit()
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                }
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.systemBackground), in: RoundedRectangle(cornerRadius: 14))
    }

    private func meta(_ r: StreamResult) -> String {
        let t = r.timings
        var s: String
        if qa.options == nil {
            s = String(format: "%.1f tok/s · %d tokens · first words %.2f s · total %.2f s",
                       t.decodeTokensPerSecond, t.generatedTokens, t.timeToFirstToken, t.total)
        } else {
            s = String(format: "%.2f s", t.total)
        }
        return s + " · \(r.frameTimes.count) past moments + now · \(r.promptTokens) prompt tokens"
    }
}

/// The question's timing breakdown (StreamSession.ask), in order.
struct TimingTable: View {
    let t: QuestionTimings

    var body: some View {
        let rows: [(String, Double)] = [
            ("Question embedding (MobileCLIP2-S0)", t.textEmbed),
            ("Frame embeddings at question time", t.frameEmbeds),
            ("Top-\(16) of \(t.candidates) stored frames", t.topK),
            ("Thumbnails (precomputed when stored)", t.thumbnails),
            ("Latest frame resize + JPEG", t.latestPrep),
            ("Latest frame vision tower", t.latestVision),
            ("Tokenize prompt", t.tokenize),
            (String(format: "Prefill %d tokens (%.0f tok/s)", t.promptTokens, t.prefillTokensPerSecond), t.prefill),
            (String(format: "Decode %d tokens (%.1f tok/s)", max(t.generatedTokens - 1, 0), t.decodeTokensPerSecond),
             t.decode),
        ]
        VStack(alignment: .leading, spacing: 2) {
            ForEach(Array(rows.enumerated()), id: \.offset) { _, r in
                HStack {
                    Text(r.0)
                    Spacer()
                    Text(String(format: "%.3f s", r.1)).monospacedDigit()
                }
            }
            HStack {
                Text("Total").bold()
                Spacer()
                Text(String(format: "%.3f s", t.total)).monospacedDigit().bold()
            }
        }
        .font(.caption2)
        .padding(.top, 4)
    }
}
