import SwiftUI

/// Question input (styled after Mobile-O's InputBar): free text, or multiple choice with up to four options.
struct QuestionBar: View {
    @Binding var question: String
    @Binding var multipleChoice: Bool
    @Binding var options: [String]
    let busy: Bool
    var focused: FocusState<Bool>.Binding
    let suggestions: [String]
    let send: () -> Void

    private var canSend: Bool {
        !question.trimmingCharacters(in: .whitespaces).isEmpty && !busy
            && (!multipleChoice || options.filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }.count >= 2)
    }

    var body: some View {
        VStack(spacing: 8) {
            if question.isEmpty && !multipleChoice {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack {
                        ForEach(suggestions, id: \.self) { s in
                            Button(s) { question = s }
                                .font(.footnote)
                                .padding(.horizontal, 10).padding(.vertical, 6)
                                .background(Color(.secondarySystemBackground), in: Capsule())
                        }
                    }
                    .padding(.horizontal, 12)
                }
            }
            if multipleChoice {
                VStack(spacing: 6) {
                    ForEach(0 ..< options.count, id: \.self) { i in
                        HStack {
                            Text(String(Character(UnicodeScalar(65 + i)!))).font(.subheadline.bold()).frame(width: 18)
                            TextField("Option", text: $options[i])
                                .textFieldStyle(.roundedBorder)
                                .font(.subheadline)
                        }
                    }
                }
                .padding(.horizontal, 12)
            }
            HStack(spacing: 10) {
                Button { withAnimation { multipleChoice.toggle() } } label: {
                    Image(systemName: multipleChoice ? "list.bullet.circle.fill" : "list.bullet.circle")
                        .font(.title2)
                }
                .accessibilityLabel("Multiple choice")
                TextField(multipleChoice ? "Question (answer: an option letter)" : "Ask about the stream…",
                          text: $question, axis: .vertical)
                    .lineLimit(1 ... 4)
                    .focused(focused)
                    .padding(.horizontal, 14).padding(.vertical, 10)
                    .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 20))
                    .onSubmit { if canSend { send() } }
                Button(action: send) {
                    if busy {
                        ProgressView().frame(width: 32, height: 32)
                    } else {
                        Image(systemName: "arrow.up.circle.fill").font(.system(size: 32))
                    }
                }
                .disabled(!canSend)
            }
            .padding(.horizontal, 12)
        }
        .padding(.vertical, 10)
        .background(.bar)
    }
}
