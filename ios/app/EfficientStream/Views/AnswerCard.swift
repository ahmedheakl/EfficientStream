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
                Text(meta(r)).font(.caption2).foregroundStyle(.secondary)
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
        var s = String(format: "%.2f s", r.seconds)
        if qa.options == nil { s = String(format: "first words %.2f s · total ", r.secondsToFirstToken) + s }
        return s + " · \(r.frameTimes.count) past moments + now · \(r.promptTokens) tokens"
    }
}
